import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

class LaplaceSolver:
    def __init__(self, w, h, ny, nz):
        self.w, self.h = w, h
        self.ny, self.nz = ny, nz
        
        # Grid setup
        self.y = np.linspace(-w/2, w/2, ny)
        self.z = np.linspace(0, h, nz)
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        
        # BC Storage
        self.dirichlet_bcs = {} 
        
        # Finite Difference Constants
        self.Ry = 1.0 / (self.dy**2)
        self.Rz = 1.0 / (self.dz**2)
        self.denom = 2 * (self.Ry + self.Rz)
        
        self.V = None # Solution storage

    def set_bc(self, side, value, start=None, end=None):
        """Sets Dirichlet BCs on a boundary segment."""
        # Helper to find indices range
        def get_range(axis_arr, s, e, max_idx):
            idx_s = 0 if s is None else np.searchsorted(axis_arr, s)
            idx_e = max_idx if e is None else np.searchsorted(axis_arr, e)
            return range(idx_s, idx_e)

        if side == 'left':
            for i in get_range(self.z, start, end, self.nz):
                self.dirichlet_bcs[(i, 0)] = value
        elif side == 'right':
            for i in get_range(self.z, start, end, self.nz):
                self.dirichlet_bcs[(i, self.ny-1)] = value
        elif side == 'bottom':
            for j in get_range(self.y, start, end, self.ny):
                self.dirichlet_bcs[(0, j)] = value
        elif side == 'top':
            for j in get_range(self.y, start, end, self.ny):
                self.dirichlet_bcs[(self.nz-1, j)] = value

    def solve(self):
        """Builds matrix and solves."""
        N = self.ny * self.nz
        A = sparse.lil_matrix((N, N))
        b = np.zeros(N)
        
        def get_k(i, j): return i * self.ny + j

        # print("Building system matrix...")
        for i in range(self.nz):
            for j in range(self.ny):
                k = get_k(i, j)
                
                if (i, j) in self.dirichlet_bcs:
                    A[k, k] = 1.0
                    b[k] = self.dirichlet_bcs[(i, j)]
                    continue
                
                # Internal/Neumann Stencil
                # Y-neighbors
                if j > 0: A[k, get_k(i, j-1)] = self.Ry
                else:     A[k, get_k(i, j+1)] += self.Ry # Neumann Mirror
                
                if j < self.ny - 1: A[k, get_k(i, j+1)] += self.Ry
                else:               A[k, get_k(i, j-1)] += self.Ry # Neumann Mirror

                # Z-neighbors
                if i > 0: A[k, get_k(i-1, j)] = self.Rz
                else:     A[k, get_k(i+1, j)] += self.Rz # Neumann Mirror
                
                if i < self.nz - 1: A[k, get_k(i+1, j)] += self.Rz
                else:               A[k, get_k(i-1, j)] += self.Rz # Neumann Mirror

                A[k, k] = -self.denom

        # print("Solving...")
        self.V = spsolve(A.tocsr(), b).reshape((self.nz, self.ny))
        return self.V

    def calculate_resistance(self, conductivity, target_voltage=None):
        """
        Calculates Resistance (Ohms) assuming 1m depth (x-direction).
        
        conductivity: S/m
        target_voltage: The voltage of the electrode to integrate current over.
                        If None, defaults to the maximum voltage found in BCs.
        """
        if self.V is None:
            raise ValueError("Run solve() before calculating resistance.")

        # 1. Identify Target Voltage
        if target_voltage is None:
            # Find max voltage in BCs
            target_voltage = max(self.dirichlet_bcs.values())
        
        # print(f"Calculating current leaving electrode at {target_voltage}V...")

        total_current = 0.0
        
        # 2. Iterate over all Dirichlet nodes to find those matching target_voltage
        # We look for nodes on the actual boundaries of the grid
        
        for (i, j), val in self.dirichlet_bcs.items():
            if not np.isclose(val, target_voltage):
                continue

            # Calculate Gradient Normal to the wall (Current Density J)
            # J = sigma * E = sigma * (dV/dn)
            # dV/dn approx (V_node - V_neighbor) / dist
            
            # Left Wall (j=0) -> Neighbor is j=1
            if j == 0:
                E_normal = (self.V[i, 0] - self.V[i, 1]) / self.dy
                area = self.dz * 1.0 # 1.0 is depth
                # Corner correction: if node is at corner, effective area is half
                if i == 0 or i == self.nz - 1: area /= 2
                total_current += conductivity * E_normal * area
                
            # Right Wall (j=end) -> Neighbor is j=end-1
            elif j == self.ny - 1:
                E_normal = (self.V[i, -1] - self.V[i, -2]) / self.dy
                area = self.dz * 1.0
                if i == 0 or i == self.nz - 1: area /= 2
                total_current += conductivity * E_normal * area

            # Bottom Wall (i=0) -> Neighbor is i=1
            elif i == 0:
                E_normal = (self.V[0, j] - self.V[1, j]) / self.dz
                area = self.dy * 1.0
                if j == 0 or j == self.ny - 1: area /= 2
                total_current += conductivity * E_normal * area

            # Top Wall (i=end) -> Neighbor is i=end-1
            elif i == self.nz - 1:
                E_normal = (self.V[-1, j] - self.V[-2, j]) / self.dz
                area = self.dy * 1.0
                if j == 0 or j == self.ny - 1: area /= 2
                total_current += conductivity * E_normal * area

        # 3. Calculate Resistance R = V / I
        # Assuming Ground is 0V. If not, R = delta_V / I
        min_voltage = min(self.dirichlet_bcs.values())
        voltage_diff = abs(target_voltage - min_voltage)
        
        resistance = voltage_diff / total_current if total_current != 0 else np.inf
        
        return resistance, total_current
        
    def calculate_current_across_z(self, z_coord, conductivity):
        """
        Integrates the vertical current density (J_z) across a horizontal line at z_coord.
        Returns: Current (Amps)
        """
        if self.V is None:
            raise ValueError("Run solve() first.")
            
        # 1. Find the closest grid index
        idx = (np.abs(self.z - z_coord)).argmin()
        
        # 2. Safety check for boundaries (cannot use central difference at very edges)
        if idx == 0:
            print("Warning: Requested z is at the bottom boundary. Using forward difference.")
            Ez = -(self.V[1, :] - self.V[0, :]) / self.dz
        elif idx == self.nz - 1:
            print("Warning: Requested z is at the top boundary. Using backward difference.")
            Ez = -(self.V[idx, :] - self.V[idx-1, :]) / self.dz
        else:
            # Central Difference: (V_up - V_down) / 2dz
            Ez = -(self.V[idx+1, :] - self.V[idx-1, :]) / (2 * self.dz)
            
        # 3. Calculate Current Density J_z (A/m^2)
        Jz = conductivity * Ez
        
        # 4. Integrate Jz along the line y = -w/2 to w/2
        # We use numpy's trapezoidal integration for robustness
        # Multiplied by 1.0 meter (depth in x-direction)
        current_z = np.trapezoid(Jz, x=self.y) * 1.0 
        
        return current_z

    def calculate_current_across_y(self, y_coord, conductivity):
        """
        Integrates the vertical current density (J_y) across a horizontal line at y_coord.
        Returns: Current (Amps)
        """
        if self.V is None:
            raise ValueError("Run solve() first.")
            
        # 1. Find the closest grid index
        idx = (np.abs(self.y - y_coord)).argmin()
       
        # 2. Safety check for boundaries (cannot use central difference at very edges)
        if idx == 0:
            print("Warning: Requested y is at the left boundary. Using forward difference.")
            Ey = -(self.V[:, 1] - self.V[:, 0]) / self.dy
        elif idx == self.nz - 1:
            print("Warning: Requested y is at the right boundary. Using backward difference.")
            Ey = -(self.V[:, idx] - self.V[:, idx-1]) / self.dy
        else:
            # Central Difference: (V_right - V_left) / 2dy
            Ey = -(self.V[:, idx+1] - self.V[:, idx-1]) / (2 * self.dy)
            
        # 3. Calculate Current Density J_y (A/m^2)
        Jy = conductivity * Ey
        
        # 4. Integrate Jy 
        current_y = np.trapezoid(Jy, x=self.z) * 1.0 
        
        return current_y


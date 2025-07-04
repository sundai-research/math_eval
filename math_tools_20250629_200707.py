"""Auto-generated math tools module"""

import collections.defaultdict
import itertools.product
import math
import math.comb
import numpy


async def NonIntersectingPathsCount(start_points: list, end_points: list) -> dict:
    from math import comb
    
    # Calculate path counts between each pair
    M = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            dx = end_points[j][0] - start_points[i][0]
            dy = end_points[j][1] - start_points[i][1]
            if dx < 0 or dy < 0:
                M[i][j] = 0
            else:
                M[i][j] = comb(dx + dy, dx)
    
    # Compute determinant
    determinant = M[0][0]*M[1][1] - M[0][1]*M[1][0]
    
    return {
        "result": determinant,
        "steps": [
            f"Matrix M: [[{M[0][0]}, {M[0][1]}], [{M[1][0]}, {M[1][1]}]]",
            f"Determinant calculation: {M[0][0]}*{M[1][1]} - {M[0][1]}*{M[1][0]} = {determinant}"
        ]
    }

async def modular_residue_pattern_generator(modulus: int, sequence_length: int, constraint_relations: list, sum_constraints: list = None) -> dict:
    from itertools import product
    from collections import defaultdict
    
    # Generate all possible residue sequences
    residues = list(range(modulus))
    all_patterns = product(residues, repeat=sequence_length)
    
    valid_patterns = []
    for pattern in all_patterns:
        # Check constraint relations
        valid = True
        for i, j in constraint_relations:
            if pattern[i] != pattern[j]:
                valid = False
                break
        if not valid:
            continue
        
        # Check sum constraints
        if sum_constraints:
            for indices, target in sum_constraints:
                if sum(pattern[i] for i in indices) % modulus != target:
                    valid = False
                    break
            if not valid:
                continue
        
        valid_patterns.append(pattern)
    
    # Calculate number of valid number selections
    num_selections = 0
    # Precompute counts of numbers in each residue class from 1-10
    residue_counts = defaultdict(int)
    for num in range(1, 11):
        residue_counts[num % modulus] += 1
    
    for pattern in valid_patterns:
        # Check if we can select distinct numbers for this pattern
        used_residues = set()
        for r in pattern:
            if r in used_residues:
                break
            used_residues.add(r)
        else:
            # Calculate combinations for this pattern
            product = 1
            for r in pattern:
                product *= residue_counts[r]
            num_selections += product
    
    return {
        'valid_patterns': valid_patterns,
        'num_valid_sequences': num_selections
    }

async def compute_tetrahedron_volume(points: list) -> dict:
    from math import sqrt
    # Extract points
    p1, p2, p3, p4 = points
    # Vectors from p1 to p2, p3, p4
    v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]
    v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]]
    v3 = [p4[0]-p1[0], p4[1]-p1[1], p4[2]-p1[2]]
    # Compute scalar triple product
    dot_product = v1[0]*(v2[1]*v3[2] - v2[2]*v3[1]) 
               - v1[1]*(v2[0]*v3[2] - v2[2]*v3[0]) 
               + v1[2]*(v2[0]*v3[1] - v2[1]*v3[0])
    volume = abs(dot_product) / 6
    return {
        "result": volume,
        "steps": [
            "Calculate vectors from p1 to p2, p3, p4",
            "Compute scalar triple product of vectors",
            "Divide absolute value by 6 to get tetrahedron volume"
        ]
    }

async def ParametricDistanceMinimizer(x_expr: str, y_expr: str, z1_real: float, z1_imag: float, z2_real: float, z2_imag: float) -> dict:
    import math
    import numpy as np
    
    def distance(theta):
        x = eval(x_expr.replace('theta', 'theta'))
        y = eval(y_expr.replace('theta', 'theta'))
        d1 = math.hypot(x - z1_real, y - z1_imag)
        d2 = math.hypot(x - z2_real, y - z2_imag)
        return d1 + d2
    
    # Grid search over theta in [0, 2π]
    theta_values = np.linspace(0, 2*np.pi, 1000)
    distances = np.array([distance(theta) for theta in theta_values])
    min_index = np.argmin(distances)
    
    return {
        'minimum_value': distances[min_index],
        'theta_at_min': theta_values[min_index],
        'steps': ['Generated parametric equations', 'Computed distances via grid search', 'Found minimum value']
    }

async def CubeLineRelationshipAnalyzer(cube_vertices: list) -> dict:
    import numpy as np
    
    # Generate all lines from vertices
    lines = []
    for i in range(len(cube_vertices)):
        for j in range(i+1, len(cube_vertices)):
            lines.append((np.array(cube_vertices[i]), np.array(cube_vertices[j])))
    
    count = 0
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            # Compute direction vectors
            dir1 = line1[1] - line1[0]
            dir2 = line2[1] - line2[0]
            
            # Check perpendicularity
            if np.dot(dir1, dir2) != 0:
                continue
            
            # Check skewness using scalar triple product
            # Line1: P + t*u, Line2: Q + s*v
            P, Q = line1[0], line2[0]
            u, v = dir1, dir2
            
            # Check if parallel
            if np.allclose(np.cross(u, v), 0):
                continue
            
            # Compute scalar triple product (Q-P) ⋅ (u × v)
            scalar_triple = np.dot(Q - P, np.cross(u, v))
            if abs(scalar_triple) > 1e-9:  # Not coplanar
                count += 1
    
    return {
        "result": count,
        "steps": [
            "Generated all lines from cube vertices",
            "Checked perpendicularity using dot product",
            "Verified skewness via scalar triple product"
        ]
    }


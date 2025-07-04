"""Auto-generated math tools module"""

import itertools.combinations
import math
import math.comb
import sympy


async def LGVNonIntersectingPaths(start1, end1, start2, end2):
    from math import comb
    
    def count_paths(s, e):
        dx = e[0] - s[0]
        dy = e[1] - s[1]
        if dx < 0 or dy < 0:
            return 0
        return comb(dx + dy, dx)
    
    a = count_paths(start1, end1)
    b = count_paths(start2, end2)
    c = count_paths(start1, end2)
    d = count_paths(start2, end1)
    
    result = a * b - c * d
    return {
        "result": result,
        "steps": [
            f"Paths from {start1} to {end1}: {a}",
            f"Paths from {start2} to {end2}: {b}",
            f"Paths from {start1} to {end2}: {c}",
            f"Paths from {start2} to {end1}: {d}",
            f"Non-intersecting paths: {a}*{b} - {c}*{d} = {result}"
        ]
    }

async def reflect_point_over_line(A: float, B: float, C: float, x0: float, y0: float) -> dict:
    """Reflects (x0, y0) over line Ax + By + C = 0"""
    denominator = A**2 + B**2
    if denominator == 0:
        return {'error': 'Invalid line equation (A and B cannot both be zero)'}
    
    numerator = A*x0 + B*y0 + C
    x_reflect = x0 - 2*A*numerator / denominator
    y_reflect = y0 - 2*B*numerator / denominator
    
    return {
        'result': (x_reflect, y_reflect),
        'steps': [
            f'Computed numerator: {numerator}',
            f'Calculated denominator: {denominator}',
            f'Computed reflected coordinates: ({x_reflect:.4f}, {y_reflect:.4f})'
        ]
    }

async def acute_triangle_counter(n: int) -> Dict:
    """Counts acute triangles in regular n-gon"""
    from itertools import combinations
    import math
    
    total = 0
    max_arc_steps = (n - 1) // 2  # Maximum steps for arc < 180 degrees
    
    for triplet in combinations(range(n), 3):
        # Sort triplet to determine consecutive arcs
        sorted_triplet = sorted(triplet)
        # Calculate steps between consecutive points
        steps = [
            sorted_triplet[1] - sorted_triplet[0],
            sorted_triplet[2] - sorted_triplet[1],
            (n - sorted_triplet[2]) + sorted_triplet[0]
        ]
        # Check if all arcs are within semicircle
        if all(step <= max_arc_steps for step in steps):
            total += 1
    
    return {
        "total_acute_triangles": total,
        "total_triangles": math.comb(n, 3),
        "probability": total / math.comb(n, 3)
    }

async def stars_and_bars(n: int, k: int) -> dict:
    """Computes the number of non-negative integer solutions to x1+...+xk=n"""
    import math
    if n < 0 or k < 0:
        return {"error": "Negative values not allowed"}
    try:
        result = math.comb(n + k - 1, k - 1)
        return {"result": result, "steps": [f"Calculating C({n}+{k}-1, {k}-1) = C({n+k-1}, {k-1})"]}
    except ValueError:
        return {"error": "Invalid input for combination"}

async def QuadraticRootIntervalChecker(a: float, b: float, c: float, p: float, q: float) -> dict:
    import math
    # Calculate discriminant
    discriminant = b**2 - 4*a*c
    # Check if discriminant is positive
    has_distinct_roots = discriminant > 0
    # Calculate function values at endpoints
    f_p = a*p**2 + b*p + c
    f_q = a*q**2 + b*q + c
    # Check if endpoints are positive (for a > 0)
    endpoint_conditions = (f_p > 0) and (f_q > 0)
    # Calculate vertex x-coordinate
    vertex_x = -b/(2*a)
    # Check if vertex is within the interval
    vertex_in_interval = (p < vertex_x < q)
    # Return results
    return {
        'has_distinct_roots': has_distinct_roots,
        'endpoint_conditions': endpoint_conditions,
        'vertex_in_interval': vertex_in_interval,
        'discriminant': discriminant,
        'f_p': f_p,
        'f_q': f_q,
        'vertex_x': vertex_x
    }

async def TrigonometricEquationSolver(equation: str, variables: list) -> dict:
    from sympy import symbols, Eq, solve
    from sympy.parsing.sympy_parser import parse_expr
    
    # Parse equation
    expr = parse_expr(equation)
    
    # Create symbols
    sym_vars = symbols(' '.join(variables))
    
    # Solve equation
    solution = solve(expr, sym_vars)
    
    return {
        'solution': str(solution),
        'steps': ['Parsed equation', 'Created symbolic variables', 'Solved equation']
    }

async def path_count_tool(start: int, end: int, steps: int) -> Dict:
    """Calculate number of paths with step constraints"""
    from math import comb
    
    # Check if end is reachable in steps
    delta = end - start
    if (steps + delta) % 2 != 0 or abs(delta) > steps:
        return {"result": 0, "steps": [f"Delta {delta} not reachable in {steps} steps"]}
    
    # Calculate number of up steps needed
    up_steps = (steps + delta) // 2
    if up_steps < 0 or up_steps > steps:
        return {"result": 0, "steps": [f"Invalid up steps {up_steps}"]}
    
    # Calculate combinations
    result = comb(steps, up_steps)
    return {"result": result, "steps": [f"Calculated C({steps}, {up_steps}) = {result}"]}

async def SymbolicGeometricEquationSolver(equations: list, variables: list) -> dict:
    from sympy import symbols, Eq, solve
    
    # Parse variables
    sym_vars = symbols(variables)
    
    # Parse equations
    parsed_eqs = []
    for eq in equations:
        parsed_eqs.append(Eq(eval(eq, {var: sym_vars[i] for i, var in enumerate(variables)})))
    
    # Solve system
    solution = solve(parsed_eqs, sym_vars)
    
    return {
        "solution": str(solution),
        "steps": ["Parsed equations:", ", ".join(equations), "Solved for variables:", ", ".join(variables)]
    }

async def SymbolicEquationSolver(equations: list, variables: list) -> dict:
    """Solves a system of symbolic equations using SymPy"""
    from sympy import symbols, Eq, solve
    
    # Convert to SymPy symbols and equations
    sym_vars = symbols(variables)
    sym_eqs = [Eq(eq[0], eq[1]) for eq in equations]
    
    # Solve the system
    solution = solve(sym_eqs, sym_vars)
    
    return {
        "solution": str(solution),
        "steps": ["Converted to SymPy symbols", "Solved system using solve()"]
    }

async def plane_cube_intersection_tool(a: float, b: float, c: float, d: float) -> Dict:
    import math
    # Define unit cube edges (12 edges)
    edges = [
        [(0,0,0), (1,0,0)], [(1,0,0), (1,1,0)], [(1,1,0), (0,1,0)], [(0,1,0), (0,0,0)],
        [(0,0,0), (0,0,1)], [(1,0,0), (1,0,1)], [(1,1,0), (1,1,1)], [(0,1,0), (0,1,1)],
        [(0,0,1), (1,0,1)], [(1,0,1), (1,1,1)], [(1,1,1), (0,1,1)], [(0,1,1), (0,0,1)]
    ]
    intersections = []
    for p1, p2 in edges:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        # Parametric line equations
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        # Plane equation: a*x + b*y + c*z + d = 0
        # Solve for t: a(x1 + t*dx) + b(y1 + t*dy) + c(z1 + t*dz) + d = 0
        A = a*dx + b*dy + c*dz
        B = a*x1 + b*y1 + c*z1 + d
        if A != 0:
            t = -B / A
            if 0 <= t <= 1:
                x = x1 + t*dx
                y = y1 + t*dy
                z = z1 + t*dz
                intersections.append((x, y, z))
    return {
        "intersections": intersections,
        "steps": ["Calculated intersections for all cube edges", "Filtered valid t values"]
    }


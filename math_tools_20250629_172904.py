"""Auto-generated math tools module"""

import math
import sympy
from typing import Dict, List


import sympy
import math

async def calculate_tangent_properties(point_coords: List[float], circle_center: List[float], circle_radius: float) -> Dict:
    """Calculates tangent length and angle between tangents from a point to a circle"""
    x, y = point_coords
    h, k = circle_center
    r = circle_radius
    
    # Calculate distance from point to circle center
    dx = x - h
    dy = y - k
    distance = sympy.sqrt(dx**2 + dy**2)
    
    # Calculate tangent length using Pythagorean theorem
    tangent_length = sympy.sqrt(distance**2 - r**2)
    
    # Calculate angle between tangents using 2*arcsin(r/distance)
    angle_radians = 2 * sympy.asin(r / distance)
    angle_degrees = sympy.deg(angle_radians)
    
    return {
        "tangent_length": str(tangent_length.simplify()),
        "angle_between_tangents": str(angle_radians.simplify()),
        "steps": [
            "Calculate distance from point to circle center: sqrt((x-h)^2 + (y-k)^2)",
            "Use Pythagorean theorem for tangent length: sqrt(d^2 - r^2)",
            "Find angle between tangents using 2*arcsin(r/d)"
        ]
    }

async def solve_functional_equation(equation: str, function_form: str) -> Dict:
    """Solve functional equations by substituting assumed forms and finding coefficients"""
    from sympy.abc import x, y
    
    # Define function form templates
    if function_form == 'linear':
        a, b = sympy.symbols('a b')
        f = a * x + b
    elif function_form == 'quadratic':
        a, b, c = sympy.symbols('a b c')
        f = a * x**2 + b * x + c
    else:
        raise ValueError(f"Unsupported function form: {function_form}")

    # Parse original equation
    lhs, rhs = equation.split('=')
    lhs = sympy.parse_expr(lhs)
    rhs = sympy.parse_expr(rhs)

    # Substitute function form
    lhs_sub = lhs.subs(f'f({x**2})', f.subs(x, x**2))
    lhs_sub = lhs_sub.subs(f'f({y**2})', f.subs(x, y**2))

    rhs_sub = rhs.subs(f'f({x+y})', f.subs(x, x+y))
    rhs_sub = rhs_sub.subs(f'f^2({x+y})', f.subs(x, x+y)**2)

    # Create equation system
    eq = sympy.Eq(lhs_sub, rhs_sub)
    solutions = sympy.solve(eq, (a, b, c) if function_form == 'quadratic' else (a, b))

    return {
        "function_form": function_form,
        "solutions": str(solutions),
        "steps": [
            f"Assumed {function_form} form: {f}",
            f"Substituted into LHS: {lhs_sub}",
            f"Substituted into RHS: {rhs_sub}",
            f"Generated equation: {eq}",
            f"Solutions found: {solutions}"
        ]
    }


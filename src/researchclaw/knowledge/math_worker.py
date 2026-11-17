"""Disposable container worker. Isolation is enforced by Docker, not parsing."""
import contextlib
import io
import json
import sys


def main():
    import sympy as sp
    request = json.loads(sys.stdin.read(20000))
    assumptions = request.get('assumptions', {})
    if not isinstance(assumptions, dict) or len(assumptions) > 20:
        raise ValueError('At most 20 symbol assumptions')
    symbols = {}
    for name, properties in assumptions.items():
        if not name.isidentifier() or not isinstance(properties, dict):
            raise ValueError('Invalid symbol assumptions')
        if any(k not in {'real','positive','negative','nonnegative','integer','nonzero'} or type(v) is not bool
               for k, v in properties.items()):
            raise ValueError('Unsupported assumption')
        symbols[name] = sp.Symbol(name, **properties)
    operation = request['operation']
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        if operation == 'python':
            namespace = {'sp': sp}
            exec(compile(request['code'], '<math-tool>', 'exec'), namespace)
            result = namespace.get('result', captured.getvalue())
            kind = 'numerical_observation'
        else:
            expression = sp.sympify(request['expression'], locals=symbols)
            variable = symbols.get(request.get('variable'), sp.Symbol(request.get('variable', 'x')))
            if operation == 'simplify': result = sp.simplify(expression)
            elif operation == 'differentiate': result = sp.diff(expression, variable)
            elif operation == 'solve': result = sp.solve(expression, variable)
            elif operation == 'evaluate': result = sp.N(expression, 30)
            else: raise ValueError('Unknown operation')
            kind = 'symbolic_check'
    print(json.dumps({'status':'completed','result':str(result)[:8000],
        'stdout':captured.getvalue()[:4000], 'kind':kind,'assumptions':assumptions,
        'precision':'SymPy exact expressions; evaluate uses 30 decimal digits; Python precision depends on code',
        'sympy':sp.__version__, 'warning':'Conditional computation only; finite samples do not prove a universal claim.'}))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(json.dumps({'status':'error','error_type':type(error).__name__,'error':str(error)[:1000]}))

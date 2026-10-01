import importlib.util
import pathlib

root = pathlib.Path(__file__).resolve().parent
script_roots = [root / 'LecturesAndTestScripts', root / 'Projects' / 'Static CT']


def load_and_run(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, 'main'):
        result = mod.main()
        print(f'{path.name}: {result}')


for folder in script_roots:
    for path in sorted(folder.glob('*.py')):
        if path.name.startswith('_'):
            continue
        load_and_run(path)
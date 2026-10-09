"""Execute notebook cells in-process through IPython, without a network kernel.

Optional helper for restricted environments. Standard Jupyter also runs the
notebook normally. Outputs below always come from actually executed cells.
"""
import argparse
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def execute(path):
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise RuntimeError(f'Notebook cell {count} failed: {error}') from error
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
    notebook.metadata['execution_method'] = 'Cells executed in order with in-process IPython; no network kernel.'
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f'Executed {count} code cells successfully; saved real outputs to {Path(path).name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', nargs='?', default='Alt-F4_FinalNotebook.ipynb')
    execute(parser.parse_args().notebook)

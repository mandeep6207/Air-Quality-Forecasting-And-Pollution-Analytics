import nbformat
from nbclient import NotebookClient, CellExecutionError
nb = nbformat.read('notebooks/air_quality_analysis.ipynb', as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name='python3')
try:
    client.execute()
    nbformat.write(nb, 'notebooks/air_quality_analysis_executed.ipynb')
    print('EXECUTION_OK')
except CellExecutionError as e:
    print('CELL_ERROR', e)
    raise

# Exploratory notebooks

Use the repository's `.venv` Python interpreter as the notebook kernel.
`ipykernel` is included in the development dependencies. No global kernel
registration is required in VS Code.

Start with a data audit and visualization. Move reusable logic into `src/forge/`
as it becomes stable. Keep raw data and generated artifacts out of notebooks
committed to Git, and label exploratory results separately from final evaluation.

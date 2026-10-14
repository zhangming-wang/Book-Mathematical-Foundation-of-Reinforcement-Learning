# Repository Guidance

- Book and slide content is checked in as PDFs; LaTeX/Beamer sources and a PDF build pipeline are absent. Code provides grid-world demos; full RL algorithm implementations linked in `Readme.md` are external.
- Prefer `Lecture slides/slidesContinuouslyUpdated/` for current slides; `Lecture slides/slidesForMyLectureVideos/` matches the recorded lectures.
- The original Python/MATLAB demos have no dependency manifests, lockfiles, automated tests, CI, or lint/typecheck configuration; use the smoke check below for original Python changes. The independent PySide6 editor has its own requirements and tests.

## Python Grid World

- Dependencies are `numpy` and `matplotlib`; `Code for grid world/README.md` documents Python 3.7-3.11.
- Run `python3 -B example_grid_world.py` from `Code for grid world/python_version/examples/`; its `sys.path.append("..")` depends on that working directory. The GUI demo renders 1,000 steps with a default 0.2-second pause and continues after `done`.
- `src/grid_world.py` imports `examples/arguments.py`, which parses process arguments immediately; unrelated CLI flags abort imports. Tuple-valued CLI flags fail because their converter is `typing.Union`; configure coordinates through that file's defaults or `GridWorld` constructor parameters.
- States are zero-based `(x, y)`; flattened index is `y * env_size[0] + x`. Plot labels and demo output add one. Python action order is down, right, up, left, stay; pass displacement tuples to `step()`.
- `reset()` returns `(state, {})`; `step(action)` returns `(state, reward, done, {})`. Boundary and forbidden-cell moves leave the agent in place with `reward_forbidden`; reaching the target sets `done`.
- Call `reset()` before stepping/rendering, then `render()` before `add_policy()` or `add_state_values()` because rendering creates their axes. Debug mode waits for keyboard input each frame; the README's claim that it saves figures is stale.

Headless smoke check from `Code for grid world/python_version/` (`-B` avoids rewriting tracked `__pycache__/*.pyc`):

```bash
MPLBACKEND=Agg python3 -B -c "from src.grid_world import GridWorld; env = GridWorld(); env.reset(); print(env.step((0, 0)))"
```

## PySide6 Grid Editor

- Independent app: repository-root `src/`, Python >= 3.10 and PySide6. From the repository root, install with `python3 -m pip install -r "src/requirements.txt"`; start with `python3 -B "src/main.py"`. Its README documents UI and result contracts.
- From `src/`, run `QT_QPA_PLATFORM=offscreen python3 -B -m unittest discover -s tests`; this workspace uses `TMPDIR=/tmp/opencode` for tests.
- `model.py` uses zero-based `(row, column)` and action order up, down, left, right, stay. Forbidden cells are enterable; transitions clamp to bounds and return destination rewards, including the current cell's reward when hitting a wall. `transition()` is a pure O(1) helper; call `config.validate()` once before transition loops.
- Both Bellman functions accept `max_count=100` (1–1000) with zero continuation value at the horizon; target cells continue acting and earning rewards, including at gamma=1. `solve_optimal_policy()` treats nonzero input probabilities as available actions, maximizes action value and splits ties equally (`abs_tol=1e-9`). The UI renders `CalculationResult` values, optimal policy and all `best_paths`.
- Optimal path search uses the final policy throughout, forbids repeated cells and stops at first target arrival within the selected step count. It returns all tied highest arrival returns; path scores exclude post-arrival rewards and may differ from start values. It does not follow a different policy for each remaining horizon; many tied actions can make enumeration expensive.
- JSON `format_version: 1` stores configuration and zoom, not calculation results or step count. Load validates before replacing the current grid; load/close guard unsaved edits. Save automatically applies pending current-cell edits, but global grid settings must be applied first. Changing the step count clears results without dirtying the grid.
- Row/column changes require confirmation and reset all cells; width/height/gamma changes preserve cells. Keep exactly one start and target; move a role to another cell before changing its former cell's type.
- Random forbidden generation replaces the whole forbidden set: exclude start/target, reset former forbidden rewards to 0, set selected rewards to -1 and preserve action probabilities. A count of 0 clears forbidden cells; generated positions persist in the existing JSON cell data.
- The board starts centered; a left click selects a cell, left-button dragging pans even when the board fits the viewport, and the mouse wheel zooms around the cursor. Keep scene padding for panning and zoom anchoring; fit-to-window recenters the board. The zoom control and saved `zoom` stay synchronized.

## MATLAB Grid World

- From `Code for grid world/matlab_version/`, run `matlab -batch "main_example"`. The actual files are `main_example.m` and `figure_plot.m`; README references to `main.m` and `figure_plot_1.m` are stale.
- Requires MATLAB R2020a+ for `exportgraphics`; policy sampling also requires `randsrc` on the MATLAB path.
- MATLAB uses one-based `(x, y)` and action order up, right, down, left, stay. Forbidden cells can be entered with a penalty, unlike the Python implementation.
- Running the demo in `matlab_version/` overwrites tracked plot PDFs: `figure_plot.m` exports fixed filenames in the working directory, including `trajectory_Bellman_Equation.pdf` twice per run.

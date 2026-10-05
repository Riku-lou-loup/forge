# EnsiCompute sessions and reusable storage

FORGE runs its detector and retrieval tools on Windows. Ollama runs inside a
Slurm allocation, reached through a relay on `nash` and an SSH tunnel. Model
requests still require `--enable-llm` in the investigation command.

## Storage lifecycle

Ollama's runtime and model files now live in `/tmp/forge-cache-<uid>` on the
allocated compute node. The directory belongs to the user and is private. A
kernel lock allows one FORGE server session per account and node to use this
cache at a time. Cache reuse does not extend a Slurm allocation or retain GPU
memory after the server stops.

The first setup downloads the runtime and any missing model. Later sessions on
the same node reuse the completed runtime and query Ollama's model list. If
`qwen3.5:4b` is installed, they skip the pull. The script does not automatically
upgrade the runtime or refresh an installed model tag. The investigation trace
records the actual Ollama version and model digest.

Free-space checks depend on the work needed. Runtime installation requires
12 GiB free, a missing model requires 5 GiB, and a session using cached files
requires 256 MiB for small session files. These are conservative thresholds,
not download sizes or disk reservations. Other users can consume free space
after the check.

The cache is local to each node. It is not school-provided permanent storage,
and administrators or system maintenance may clear `/tmp`. A different node
may need its own download. Do not put this cache back in the quota-limited home
directory without confirming an appropriate storage allocation.

## Preserve a running legacy session

The earlier script downloaded into a temporary directory and deleted it at
shutdown. Before stopping such a session, run the following command from the
repository root in another Windows terminal, substituting its current job ID:

```powershell
.\.venv\Scripts\python.exe scripts\ensicompute_session.py --user YOUR_LOGIN --preserve-job JOB_ID
```

This starts a CPU-only step inside that existing allocation. It identifies the
user-owned legacy Ollama directory from processes in the same job and checks
that Ollama lists the completed model. It creates hard links to the runtime and
model files in the cache, without another download or a second copy of the
large file contents. Deleting the original session directory later does not
remove the cached links. The command refuses to overwrite an existing cache.

Wait for `Preserved runtime and model` before stopping the old session. The
command does not stop its server or cancel its allocation. If the old job has
already ended, this live-session preservation command cannot attach to it.
It does not automatically adopt arbitrary leftover directories.

## Start and connect

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\ensicompute_session.py --user YOUR_LOGIN --allocate --start-ollama --minutes 60
```

For a warm cache, expect `Reusing cached Ollama runtime` and
`Reusing cached model ... no pull requested`, followed by an `ollama_ready`
event containing the job ID, node and cache path. Keep the terminal open.

In a second terminal, attach the relay to the newly reported job ID:

```powershell
.\.venv\Scripts\python.exe scripts\ensicompute_session.py --user YOUR_LOGIN --connect-job JOB_ID
```

After `Relay ready`, run the investigation in a third terminal:

```powershell
.\.venv\Scripts\python.exe -m forge.agents.local_demo --enable-llm --backend-llm ensicompute --recording valve1/1 --export
```

Authentication remains in OpenSSH. The script stores no password. This workflow
still uses separate allocation and relay terminals; it is not a single-command
launcher for the entire application.

## Shutdown and limits

Stop the relay terminal first, then press Ctrl+C in the allocation terminal.
The updated supervisor requests a cooperative stop inside its job and gives
the compute worker time to close the server and remove its own session folder.
Runtime and model files stay in the cache. If cooperative shutdown fails, the
supervisor falls back to signalling the compute step and requesting job
cancellation. It reports failures rather than claiming cleanup was confirmed.

The worker removes session files in a `finally` block even if server shutdown
raises an error. Forced process termination, node failure or loss of access can
still prevent Python cleanup. A small stale session folder or an interrupted
installation stage may remain. The script does not sweep other jobs' directories
or silently delete the reusable cache.

Verify job release on `nash` with `squeue -u YOUR_LOGIN`. An empty queue confirms
there are no active jobs for that account; it does not by itself verify disk
cleanup. Cache paths are printed so they can be inspected separately.

## Verification

Before this cache change, live checks verified allocation, cancellation,
Ollama startup, relay access, and a full FORGE investigation through EnsiCompute.
The cache change has offline tests for repeated reuse without downloads,
missing-model pulls, failed installation cleanup, hard-link preservation,
directory ownership checks, lock contention, and cooperative shutdown ordering.
Linux permission and locking behavior is simulated in the Windows test suite.
Live cache preservation and reuse still need verification on the cluster.

The implementation follows [Slurm's job-step interface](https://slurm.schedmd.com/srun.html)
and Ollama's [model-storage configuration](https://docs.ollama.com/faq).

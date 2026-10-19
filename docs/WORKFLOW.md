# Beginner Workflow: From VS Code to Submission

## Phase 1 — Understand and plan
- Read the official problem statement.
- Identify required modules: detection, embeddings, tracking, registration, event logging, database, config, and documentation.
- Use `docs/AI_PLANNING.md` as the initial plan.
- Divide work across three teammates:
  - Teammate A: environment, model setup, detection/embedding pipeline.
  - Teammate B: database, event logging, tracker.
  - Teammate C: integration tests, README, demo recording, submission checklist.
- All teammates should review the full system and be able to explain the code.

## Phase 2 — Set up
1. Install Python 3.10/3.11, Git, and VS Code.
2. Open the extracted project folder in VS Code.
3. Create and activate a virtual environment.
4. Run `pip install -r requirements.txt`.
5. Put a compatible face-specific YOLO weights file in `models/`.
6. Save the organizer-provided video as `sample.mp4` in the project root.

## Phase 3 — Run and debug
1. Check `config.json` paths.
2. Run `python -m src.main`.
3. If the YOLO model path error appears, confirm the `.pt` file exists and is a face detector.
4. If package installation fails, record the full error and Python version before changing versions.
5. Confirm a window opens and face boxes appear.
6. Check `logs/events.log`.
7. Check `logs/entries/` and `logs/exits/`.
8. Query `data/visitors.db` using the command in README.
9. Test a short video first; do not wait until the deadline to run the full clip.

## Phase 4 — Validate
- Manually count unique faces in a short segment.
- Verify identity reuse for repeated appearances.
- Verify one entry and one exit per track.
- Verify each event image exists.
- Record false matches, duplicate registrations, missed faces, and runtime performance.
- Tune configuration based on evidence; document what changed.

## Phase 5 — GitHub
Run these from the project root:

```bash
git init
git add .
git status
git commit -m "Initial face tracker implementation"
```

Create an empty repository on GitHub, then use the repository URL GitHub provides:

```bash
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Before pushing, check `git status` and ensure private face images, the database, private videos, credentials, and unlicensed model weights are not being committed. `.gitignore` excludes several generated files, but always verify.

For team collaboration, agree on one person responsible for merging the final working version. Use branches for separate tasks and review changes before merging.

## Phase 6 — Demo video
Record a short walkthrough:
1. Problem and architecture.
2. Configuration and run command.
3. Sample video showing face boxes, IDs, and unique count.
4. `events.log`, entry/exit image folders, and SQLite query results.
5. Limitations, compute measurements, and next improvements.

Upload the video to an allowed Loom/YouTube location and add the link to README. Do not show private faces or credentials without authorization.

## Phase 7 — Final submission
- [ ] App runs from a clean setup.
- [ ] Required code and documentation are in GitHub.
- [ ] README contains setup, assumptions, config, architecture, AI plan, and demo link.
- [ ] Sample output includes permitted logs/images/database evidence.
- [ ] Required Katomaran line is at the bottom of README.
- [ ] Repository URL is submitted in the official form before the stated deadline: 12 PM Monday, October 19, 2026.
- [ ] Each teammate can explain their work and the full pipeline.

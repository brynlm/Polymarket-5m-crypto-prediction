"""Scheduled entry point: retrain the XGB model and deploy it only if it looks sane.

Run daily by .github/workflows/scheduled_retrain.yml. Calls
retrain.train_and_save() — the same code path a manual `python3 retrain.py`
run uses — then gates on the returned diagnostics before touching git. On
failure (bad diagnostics or any exception, including retrain.py's own
feat_cols regression guard), this exits non-zero without committing or
pushing, so the currently-deployed model is left untouched.
"""
import os
import re
import subprocess
import sys

import retrain

MIN_COMBINED_ROWS  = 20_000
COVERAGE_80_RANGE  = (0.65, 0.90)
MIN_DIRECTIONAL_ACC = 0.52

MODEL_FILES = ['xgb_qreg_5s.joblib', 'xgb_qreg_5s_meta.json']
BOT_AUTHOR_NAME = 'retrain-bot'


def check_diagnostics(diag: dict) -> list[str]:
    """Return a list of human-readable failure reasons; empty list = pass."""
    failures = []

    if diag['n_combined_rows'] < MIN_COMBINED_ROWS:
        failures.append(
            f"n_combined_rows={diag['n_combined_rows']} < {MIN_COMBINED_ROWS}"
        )

    for mkt, stats in diag['markets'].items():
        lo, hi = COVERAGE_80_RANGE
        if not (lo <= stats['coverage_80'] <= hi):
            failures.append(
                f"{mkt}: coverage_80={stats['coverage_80']:.4f} outside [{lo}, {hi}]"
            )
        if stats['directional_acc'] < MIN_DIRECTIONAL_ACC:
            failures.append(
                f"{mkt}: directional_acc={stats['directional_acc']:.4f} "
                f"< {MIN_DIRECTIONAL_ACC}"
            )

    return failures


def _origin_slug() -> str:
    """Extract 'OWNER/REPO' from the origin remote, https or git@ form."""
    url = subprocess.run(
        ['git', 'remote', 'get-url', 'origin'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    m = re.search(r'[:/]([^/:]+/[^/]+?)(\.git)?$', url)
    if not m:
        raise ValueError(f"Couldn't parse owner/repo from origin URL: {url}")
    return m.group(1)


def _push_target() -> str:
    """Where to push. GitHub Actions' actions/checkout (given `permissions:
    contents: write`) already configures the checkout with push-capable
    credentials, so 'origin' just works there. GITHUB_PUSH_TOKEN is an
    optional escape hatch for running this outside GitHub Actions, where no
    such credential is pre-configured.
    """
    token = os.environ.get('GITHUB_PUSH_TOKEN')
    if not token:
        return 'origin'
    return f'https://x-access-token:{token}@github.com/{_origin_slug()}.git'


def _last_commit_author() -> str:
    return subprocess.run(
        ['git', 'log', '-1', '--pretty=%an'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def commit_and_push(diag: dict) -> None:
    """Commit the retrained model and push.

    xgb_qreg_5s.joblib is a ~3.5MB pickled binary that doesn't diff well, so
    a fresh commit every day would add ~1.2GB/year of near-duplicate blobs
    to the repo's history. Instead, when the current tip of main is already
    a bot-authored automated-retrain commit, amend it in place and
    force-push — one rolling commit represents "the current auto-retrained
    model" rather than an ever-growing chain. A manual commit (like a
    human-run retrain) is never amended — only a prior bot commit is, so a
    fresh commit is created on top of it instead.
    """
    push_target = _push_target()

    subprocess.run(['git', 'config', 'user.email', 'retrain-bot@users.noreply.github.com'], check=True)
    subprocess.run(['git', 'config', 'user.name', BOT_AUTHOR_NAME], check=True)
    subprocess.run(['git', 'add', *MODEL_FILES], check=True)

    staged = subprocess.run(['git', 'diff', '--cached', '--quiet'])
    if staged.returncode == 0:
        print("No changes to model files — nothing to commit, skipping push.")
        return

    summary = ", ".join(
        f"{mkt} cov80={s['coverage_80']:.3f} dir_acc={s['directional_acc']:.3f}"
        for mkt, s in diag['markets'].items()
    )
    commit_msg = f"automated retrain ({diag['trained_at']}): {summary}"

    if _last_commit_author() == BOT_AUTHOR_NAME:
        subprocess.run(['git', 'commit', '--amend', '-m', commit_msg], check=True)
        # --force-with-lease refuses to push if origin/main moved since our
        # checkout (e.g. a manual push landed in between) rather than blindly
        # overwriting it, unlike a plain --force.
        subprocess.run(['git', 'push', '--force-with-lease', push_target, 'HEAD:main'], check=True)
        print("Amended previous automated-retrain commit and force-pushed.")
    else:
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        subprocess.run(['git', 'push', push_target, 'HEAD:main'], check=True)
        print("Tip of main was a manual commit — created a new commit and pushed.")


def main() -> None:
    diag = retrain.train_and_save()

    failures = check_diagnostics(diag)
    if failures:
        print("Safety checks FAILED — not deploying:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print("Safety checks passed.")
    commit_and_push(diag)


if __name__ == "__main__":
    main()

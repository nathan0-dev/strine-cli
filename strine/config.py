from dotenv import find_dotenv, load_dotenv

DEFAULT_MODEL = "claude-sonnet-5"


class MissingAPIKeyError(RuntimeError):
    pass


def load_project_env() -> None:
    """Loads the .env from the directory where the user RAN strine.

    usecwd=True: resolves .env starting from cwd, not from where the
    package is installed — without this, a .env near the source code
    (e.g. a dev checkout) would leak into any execution, in any directory.
    override=True: the project's .env takes priority over a variable
    already exported in the shell (avoids silently using an old/wrong key).
    """
    load_dotenv(find_dotenv(usecwd=True), override=True)

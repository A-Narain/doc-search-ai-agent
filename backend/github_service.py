from github import Github
from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")


def upload_to_github(filepath, filename):

    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO)

    with open(filepath, "rb") as file:
        content = file.read()

    github_path = f"documents/{filename}"

    try:
        # File already exists → update it
        existing_file = repo.get_contents(github_path)

        repo.update_file(
            path=github_path,
            message=f"Update {filename}",
            content=content,
            sha=existing_file.sha
        )

        print(f"Updated {filename} on GitHub")

    except:
        # File does not exist → create it
        repo.create_file(
            path=github_path,
            message=f"Upload {filename}",
            content=content
        )

        print(f"Created {filename} on GitHub")
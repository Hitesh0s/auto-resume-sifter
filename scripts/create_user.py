"""
User provisioning CLI for Auto Resume Sifter.

Usage:
    py -3.11 scripts/create_user.py --add <username> --password <password> [--name "Full Name"] [--email user@co.com]
    py -3.11 scripts/create_user.py --list
    py -3.11 scripts/create_user.py --delete <username>
"""

import argparse
import pathlib
import sys

import bcrypt
import yaml
from yaml.loader import SafeLoader

_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml"


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {
            "credentials": {"usernames": {}},
            "cookie": {
                "expiry_days": 7,
                "key": "ars_cookie_secret_key_change_in_prod",
                "name": "ars_session",
            },
        }
    with open(_CONFIG_PATH) as f:
        return yaml.load(f, Loader=SafeLoader)


def _save_config(config: dict) -> None:
    with open(_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def add_user(username: str, password: str, name: str, email: str) -> None:
    config = _load_config()
    users = config["credentials"]["usernames"]
    if username in users:
        print(f"User '{username}' already exists. Use --delete first to replace.")
        sys.exit(1)
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()
    users[username] = {"name": name, "email": email, "password": hashed}
    _save_config(config)
    print(f"User '{username}' ({name}) added successfully.")


def list_users() -> None:
    config = _load_config()
    users = config["credentials"]["usernames"]
    if not users:
        print("No users configured.")
        return
    print(f"{'Username':<20} {'Name':<25} {'Email'}")
    print("-" * 65)
    for uname, info in users.items():
        print(f"{uname:<20} {info.get('name',''):<25} {info.get('email','')}")


def delete_user(username: str) -> None:
    config = _load_config()
    users = config["credentials"]["usernames"]
    if username not in users:
        print(f"User '{username}' not found.")
        sys.exit(1)
    del users[username]
    _save_config(config)
    print(f"User '{username}' deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Auto Resume Sifter users")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--add", metavar="USERNAME", help="Add a new user")
    group.add_argument("--list", action="store_true", help="List all users")
    group.add_argument("--delete", metavar="USERNAME", help="Delete a user")
    parser.add_argument("--password", metavar="PASSWORD", help="Password for --add")
    parser.add_argument("--name", metavar="NAME", default="", help="Display name for --add")
    parser.add_argument("--email", metavar="EMAIL", default="", help="Email for --add")

    args = parser.parse_args()

    if args.add:
        if not args.password:
            parser.error("--password is required with --add")
        add_user(args.add, args.password, args.name or args.add, args.email)
    elif args.list:
        list_users()
    elif args.delete:
        delete_user(args.delete)


if __name__ == "__main__":
    main()

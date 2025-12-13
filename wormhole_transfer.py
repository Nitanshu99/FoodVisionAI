#!/usr/bin/env python3
"""
Simple file/directory transfer script using magic-wormhole
Usage:
    Send file:      python wormhole_transfer.py -s -f /path/to/file
    Send directory: python wormhole_transfer.py -s -d /path/to/directory
    Receive:        python wormhole_transfer.py -r
"""

import argparse
import sys
import os
from subprocess import run, CalledProcessError

def send_file(path):
    """Send a file using wormhole"""
    if not os.path.exists(path):
        print(f"Error: '{path}' does not exist")
        sys.exit(1)
    
    if not os.path.isfile(path):
        print(f"Error: '{path}' is not a file. Use -d for directories")
        sys.exit(1)
    
    print(f"Sending file: {path}")
    try:
        run(["wormhole", "send", path], check=True)
    except CalledProcessError as e:
        print(f"Error sending file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'wormhole' command not found. Install it with: pip install magic-wormhole")
        sys.exit(1)

def send_directory(path):
    """Send a directory using wormhole"""
    if not os.path.exists(path):
        print(f"Error: '{path}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a directory. Use -f for files")
        sys.exit(1)
    
    print(f"Sending directory: {path}")
    try:
        run(["wormhole", "send", "--code-length=2", path], check=True)
    except CalledProcessError as e:
        print(f"Error sending directory: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'wormhole' command not found. Install it with: pip install magic-wormhole")
        sys.exit(1)

def receive():
    """Receive a file or directory using wormhole"""
    print("Ready to receive. Waiting for wormhole code...")
    try:
        run(["wormhole", "receive"], check=True)
    except CalledProcessError as e:
        print(f"Error receiving: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'wormhole' command not found. Install it with: pip install magic-wormhole")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Simple file/directory transfer using magic-wormhole",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Send a file:       %(prog)s -s -f document.pdf
  Send a directory:  %(prog)s -s -d my_folder
  Receive:           %(prog)s -r
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-s', '--send', action='store_true',
                           help='Send mode')
    mode_group.add_argument('-r', '--receive', action='store_true',
                           help='Receive mode')
    
    # Path selection (only for send mode)
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument('-f', '--file', type=str, metavar='PATH',
                           help='Send a file')
    path_group.add_argument('-d', '--directory', type=str, metavar='PATH',
                           help='Send a directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.send and not (args.file or args.directory):
        parser.error("Send mode requires either -f/--file or -d/--directory")
    
    if args.receive and (args.file or args.directory):
        parser.error("Receive mode does not accept -f/--file or -d/--directory")
    
    # Execute the appropriate action
    if args.send:
        if args.file:
            send_file(args.file)
        elif args.directory:
            send_directory(args.directory)
    elif args.receive:
        receive()

if __name__ == "__main__":
    main()

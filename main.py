import argparse
from record import record
from crop import crop_faces
from train import train
from live import live

# NOTE: This is the main file, you propably don't need to change anything here

# Setup an argument parser for control via command line
parser = argparse.ArgumentParser(
    prog="Face Recognition Pipeline",
    description="A machine learning pipeline for face recognition",
    epilog="Students project",
)

parser.add_argument("mode", choices=["record", "crop", "train", "live"])
parser.add_argument("-f", "--folder")
parser.add_argument("-b", "--border", action="store", default=0.1)
parser.add_argument("-s", "--split", action="store", default=0.2)
parser.add_argument("-e", "--epochs", action="store", default=30)

# Parse the arguments from the command line
args = parser.parse_args()

# Switch control flow based on arguments
if args.mode == "record":
    record(args)

if args.mode == "crop":
    crop_faces(args)

if args.mode == "train":
    train(args)

if args.mode == "live":
    live(args)

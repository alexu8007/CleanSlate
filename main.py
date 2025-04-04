#!/usr/bin/env python
"""
CleanSlate - Enterprise-grade data cleaning solution.

This module provides a command-line interface for the CleanSlate application.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from cleanslate import CleanSlate, Config
from cleanslate.ui.run_ui import run_ui

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cleanslate.main")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CleanSlate - Enterprise-grade data cleaning solution.")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean a dataset")
    clean_parser.add_argument("input", help="Input file path")
    clean_parser.add_argument("-o", "--output", help="Output file path")
    clean_parser.add_argument("-p", "--pipeline", help="Pipeline file path")
    clean_parser.add_argument("-c", "--config", help="Configuration file path")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a dataset")
    analyze_parser.add_argument("input", help="Input file path")
    analyze_parser.add_argument("-o", "--output", help="Output file path for analysis results")
    analyze_parser.add_argument("-c", "--config", help="Configuration file path")
    
    # Score command
    score_parser = subparsers.add_parser("score", help="Score the quality of a dataset")
    score_parser.add_argument("input", help="Input file path")
    score_parser.add_argument("-o", "--output", help="Output file path for quality score results")
    score_parser.add_argument("-c", "--config", help="Configuration file path")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the web UI")
    ui_parser.add_argument("-p", "--port", type=int, default=8501, help="Port to run the UI on")
    ui_parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    ui_parser.add_argument("-c", "--config", help="Configuration file path")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    return parser.parse_args()


def command_clean(args):
    """Run the clean command."""
    # Initialize CleanSlate with configuration if provided
    cleaner = CleanSlate(args.config)
    
    # Load the dataset
    logger.info(f"Loading dataset from {args.input}")
    dataset = cleaner.load_data(args.input)
    
    # Clean the dataset
    if args.pipeline:
        # Load pipeline from file
        logger.info(f"Loading pipeline from {args.pipeline}")
        pipeline = cleaner.create_pipeline()
        pipeline = pipeline.load(args.pipeline, cleaner.config)
        
        # Clean using the loaded pipeline
        logger.info("Cleaning dataset with custom pipeline")
        cleaned_dataset = cleaner.clean(dataset, pipeline.id)
    else:
        # Clean using automatic pipeline
        logger.info("Cleaning dataset with automatic pipeline")
        cleaned_dataset = cleaner.clean(dataset)
    
    # Calculate quality improvement
    before_scores = cleaner.score(dataset)
    after_scores = cleaner.score(cleaned_dataset)
    
    # Print quality improvement
    logger.info("Quality improvement:")
    logger.info(f"  Completeness: {before_scores['completeness']:.2f} -> {after_scores['completeness']:.2f} ({after_scores['completeness'] - before_scores['completeness']:+.2f})")
    logger.info(f"  Consistency: {before_scores['consistency']:.2f} -> {after_scores['consistency']:.2f} ({after_scores['consistency'] - before_scores['consistency']:+.2f})")
    logger.info(f"  Validity: {before_scores['validity']:.2f} -> {after_scores['validity']:.2f} ({after_scores['validity'] - before_scores['validity']:+.2f})")
    logger.info(f"  Uniqueness: {before_scores['uniqueness']:.2f} -> {after_scores['uniqueness']:.2f} ({after_scores['uniqueness'] - before_scores['uniqueness']:+.2f})")
    logger.info(f"  Overall: {before_scores['overall']:.2f} -> {after_scores['overall']:.2f} ({after_scores['overall'] - before_scores['overall']:+.2f})")
    
    # Export cleaned dataset if output path is provided
    if args.output:
        logger.info(f"Exporting cleaned dataset to {args.output}")
        cleaner.export(cleaned_dataset, args.output)
    
    logger.info("Done!")


def command_analyze(args):
    """Run the analyze command."""
    # Initialize CleanSlate with configuration if provided
    cleaner = CleanSlate(args.config)
    
    # Load the dataset
    logger.info(f"Loading dataset from {args.input}")
    dataset = cleaner.load_data(args.input)
    
    # Detect anomalies
    logger.info("Detecting anomalies")
    anomalies = cleaner.detect_anomalies(dataset)
    
    # Print anomaly summary
    logger.info("Anomaly detection results:")
    logger.info(f"  Total anomalies: {anomalies['summary']['total_anomalies']}")
    logger.info(f"  Anomaly percentage: {anomalies['summary']['anomaly_percentage']:.2f}%")
    logger.info(f"  Total missing values: {anomalies['summary']['total_missing']}")
    logger.info(f"  Missing percentage: {anomalies['summary']['missing_percentage']:.2f}%")
    
    # Print column-specific anomalies
    if anomalies["columns"]:
        logger.info("Anomalies by column:")
        for col, col_anomalies in anomalies["columns"].items():
            logger.info(f"  {col}: {col_anomalies['count']} anomalies ({col_anomalies['percentage']:.2f}%)")
    
    # Export analysis results if output path is provided
    if args.output:
        import json
        
        logger.info(f"Exporting analysis results to {args.output}")
        with open(args.output, "w") as f:
            json.dump(anomalies, f, indent=2)
    
    logger.info("Done!")


def command_score(args):
    """Run the score command."""
    # Initialize CleanSlate with configuration if provided
    cleaner = CleanSlate(args.config)
    
    # Load the dataset
    logger.info(f"Loading dataset from {args.input}")
    dataset = cleaner.load_data(args.input)
    
    # Score the dataset
    logger.info("Calculating quality scores")
    scores = cleaner.score(dataset)
    
    # Print scores
    logger.info("Quality scores:")
    logger.info(f"  Completeness: {scores['completeness']:.2f}")
    logger.info(f"  Consistency: {scores['consistency']:.2f}")
    logger.info(f"  Validity: {scores['validity']:.2f}")
    logger.info(f"  Uniqueness: {scores['uniqueness']:.2f}")
    logger.info(f"  Overall: {scores['overall']:.2f}")
    
    # Export scores if output path is provided
    if args.output:
        import json
        
        logger.info(f"Exporting quality scores to {args.output}")
        with open(args.output, "w") as f:
            json.dump(scores, f, indent=2)
    
    logger.info("Done!")


def command_ui(args):
    """Run the UI command."""
    logger.info(f"Launching CleanSlate UI on port {args.port}")
    run_ui(port=args.port, debug=args.debug)


def command_version():
    """Run the version command."""
    from cleanslate import __version__
    print(f"CleanSlate version {__version__}")


def main():
    """Main entry point for the CleanSlate CLI."""
    args = parse_args()
    
    # Handle commands
    if args.command == "clean":
        command_clean(args)
    elif args.command == "analyze":
        command_analyze(args)
    elif args.command == "score":
        command_score(args)
    elif args.command == "ui":
        command_ui(args)
    elif args.command == "version":
        command_version()
    else:
        # No command specified, show help
        import sys
        from argparse import ArgumentParser
        
        parser = ArgumentParser(description="CleanSlate - Enterprise-grade data cleaning solution.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

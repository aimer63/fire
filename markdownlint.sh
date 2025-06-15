#!/bin/bash

# Format all Markdown files with prettier and markdownlint-cli2

prettier --write --prose-wrap always --print-width 100 "**/*.md"

markdownlint-cli2 -- fix "**/*.md"

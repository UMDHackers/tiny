#!/bin/bash
# Usage: ./ralph-once.sh "Next task from PRD"
# This script assumes you are using an agentic CLI (like Claude Code or Gemini CLI)
agent-cli --prompt "Read PRD.md and progress.txt. Complete the next logical task. Run tests. If pass, update progress.txt and commit with a clear message."

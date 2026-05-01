# Project Specification: Animal Avatar Orchestrator

## Description
A local pipeline designed to identify animals from photos and generate a stylized 2D avatar.

## Components
1. **Vision Agent**: Powered by Ollama (LLaVA model). It performs feature extraction.
2. **Artist Agent**: Powered by Stable Diffusion. It generates the final graphic.
3. **Orchestrator**: Main logic that coordinates the data flow between agents.

## Directory Map
- `/input`: Place source images here.
- `/output`: Resulting avatars are saved here.
- `/src`: Application source code.

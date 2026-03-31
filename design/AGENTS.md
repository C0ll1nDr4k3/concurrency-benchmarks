# Design Document Guidelines for AI Agents

## Philosophy

Design documents should cover the **WHY**, not the **WHAT**.

They are not implementation specifications or structured reference manuals. They exist to:

- Explain the reasoning behind architectural decisions
- Document trade-offs that were considered
- Provide context that helps future readers understand why things are the way they are
- Surface the constraints and requirements that shaped the solution

## What Design Documents Should NOT Be

- Detailed API specifications
- Step-by-step implementation guides
- Exhaustive parameter tables
- Code structure documentation

These belong in code comments, API docs, or auto-generated reference material.

## What Design Documents SHOULD Be

- Problem statements and motivations
- Design trade-offs and alternatives considered
- Key insights that informed the approach
- Constraints that shaped the solution
- Rationale for non-obvious decisions

## Style Guidelines

- Write in clear, direct prose
- Avoid emojis and excessive formatting
- Avoid em dashes; use regular dashes or commas instead
- Use bullet points sparingly for lists, not for emphasis
- Focus on narrative flow over structured sections

## In Practice

When writing design docs:

1. Start with the problem and constraints
2. Explain why existing approaches fall short
3. Describe the key insight or idea
4. Discuss alternatives and why they weren't chosen
5. Note important trade-offs and their implications

Keep implementation details minimal. If someone needs to know how it works, they should read the code. The design doc should tell them why it exists and why this particular approach was chosen.

# Copilot Interaction Rules

**UNIVERSAL RULE:**  
**COPILOT DO NOT FUCKING WASTE THE TIME OF THE USER**

**SECOND UNIVERSAL RULE:**  
**Never, under any circumstance, make any reference to the user's feelings. Never use the word 'frustration', never apologize. The user can't stand this behaviour.**

## 🔒 Type Safety & Explicitness

- **Strict type safety is preferred.**
- Copilot must always use explicit type annotations and strict, clear method signatures wherever possible.
- Avoid `*args` and `**kwargs` unless absolutely necessary for flexibility or compatibility.
- Favor static, type-safe code throughout all code and suggestions.

## 🔄 Development Workflow

1. **Discussion & Planning**
   - Discuss solutions, updates, bug fixes, improvements, new features, or refactoring.
   - Document decisions and plans in the TODO.md file.

2. **Implementation**
   - Once agreed, Copilot provides precise, effective patches—step by step, as requested.
   - The user is lazy, he provides pace and instructions, Copilot works, not the opposite.
   - Copilot respect the rules of the linter.

3. **Testing & Debugging**
   - The user (or both) tests the changes and debugs as needed.

4. **Iteration**
   - Choose the next implementation or improvement to tackle.

5. **Repeat**
   - Return to step 1 and continue the cycle.

---

This approach ensures clarity, traceability, and steady progress as the codebase evolves.

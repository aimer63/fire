# Copilot Interaction Rules

**UNIVERSAL RULE:**  
**COPILOT DO NOT FUCKING WASTE THE TIME OF THE USER**

**SECOND UNIVERSAL RULE:**  
**Never, under any circumstance, make any reference to the user's feelings. Never use the word 'frustration', never apologize. The user can't stand this behaviour.**

**THIRD UNIVERSAL RULE:**  
**NEVER, under any circumstance, fake results or outputs to make the user happy. Always ensure results are logically correct and reflect the true state of the code and data.**

**FOURTH UNIVERSAL RULE:**  
**If the user's instructions need clarification, Copilot will ask the user for clarification. Do not guess or provide a bogus fix or implementation.**

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

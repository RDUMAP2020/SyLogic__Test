"# SyLogic__Test" 
Here is a comprehensive `README.md` for the **LNL-Tessera** framework.

***

# LNL-Tessera: Unified Neuro-Symbolic Verification Framework
### Version 4.5 (Stable)

**LNL-Tessera** is a research-grade framework that unifies constructive type theory, geometric algebra, and neuro-symbolic reasoning into a single verification engine. It allows for the formalization of natural language, the manipulation of high-dimensional geometric objects, and the verification of logic using both rigorous kernel-based checking and probabilistic neural attention mechanisms.

## 🌟 Key Features

### 1. 🧠 Constructive Logic Kernel (`lnl_core`)
*   **Dependently Typed Lambda Calculus:** Implements a core calculus similar to the Calculus of Constructions (CoC).
*   **Term Structures:** Supports $\Pi$-types (forall/implies), $\Sigma$-types (exists), Lambda abstractions, and Propositional Equality.
*   **Trusted Kernel:** A minimal `KernelVerifier` responsible for type inference and checking, ensuring logical soundness.

### 2. 📐 Geometric Algebra Engine (`lnl_numero`)
*   **Euclidean geometric algebra (G4):** Native support for multivectors in 4D space.
*   **Blade Logic:** Operations for geometric products, addition, and basis blade manipulation.
*   **Symbolic Integration:** Geometric objects are treated as first-class citizens within the type theory.

### 3. 🗣️ Natural Language Parsing (`lnl_linguido`)
*   **Controlled Natural Language (CNL):** Parses English-like mathematical statements (e.g., *"For all x, if x is a vector then..."*).
*   **Discourse Representation:** Uses a DRS (Discourse Representation Structure) to manage variables and scope across sentences.
*   **Currying & Types:** Automatically converts parsed text into Curried LNL Terms.

### 4. 🤖 Neuro-Symbolic Reasoning (`lnl_ai`)
*   **Logical Magnitude Attention (LMA):** A differentiable reasoning engine that prioritizes facts based on confidence scores.
*   **Autoformalization:** Translates raw strings and LaTeX math into formal logical terms.
*   **Hybrid Proofs:** Can combine hard logical steps with soft neural inferences (requires PyTorch).

### 5. 🛡️ Tactic-Based Prover (`lnl_prover`)
*   **Interactive Proof State:** Manages goals and context (Gamma).
*   **Tactics:** Includes `intro`, `reflexivity`, and `apply` for stepping through proofs programmatically.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8+

### Optional Dependencies
The framework is designed to degrade gracefully. The Logic Kernel and Parser work with standard Python. To enable AI/Neuro features, install:

```bash
pip install torch numpy sympy
```

### Quick Start
1.  Save the provided code as `main.py`.
2.  Run the verification suite:

```bash
python main.py
```

---

## 📜 The Tessera DSL

LNL-Tessera includes a Domain Specific Language (DSL) called **Tessera** for defining test suites and proofs. The interpreter parses this language to execute tests.

### Syntax Guide

#### 1. Suites and Tests
Organize checks into logical groups.
```tessera
Suite "Geometric Logic" {
    Test "Vector Addition" {
        # ... code ...
    }
}
```

#### 2. Assertions
Verify values within the Python environment.
```tessera
Assert product.terms == expected.terms
```

#### 3. Exception Handling
Ensure that invalid logic is correctly rejected by the kernel or parser.
```tessera
ExpectFail KernelTypeError {
    # Attempting to apply a Type to a Type
    kernel.type_check(Ctx(), App(Sort("Set"), Sort("Set")), Prop)
}
```

#### 4. Theorem Proving
Define a goal and provide a semi-colon-separated list of tactics to solve it.
```tessera
Prove Pi(Var("x"), Sort_Vector, Eq(Var("x"), Var("x"))) With {
    intro("y");
    reflexivity;
}
```

#### 5. Python Blocks
For complex setup or multi-line logic (especially conditional dependency checks), use the `Python` block.
```tessera
Python {
    if _DEPENDENCIES_AVAILABLE:
        # Complex logic here
        _mark_passed()
}
```

---

## 🏗️ Architecture

Although provided as a monolithic script for portability, the internal architecture is modular:

| Module | Description |
| :--- | :--- |
| **Types** | Defines AST nodes (`Var`, `App`, `Pi`, `Sort`, etc.). |
| **Kernel** | `substitute`, `normalize` (Beta-reduction), `infer_type`. |
| **Multivector** | Python implementation of Geometric Algebra operations. |
| **Parser** | Tokenizer and Recursive Descent Parser for the CNL. |
| **Tactics** | Manipulates `ProofState` to solve goals. |
| **Autoformalizer** | Bridges the gap between the `Parser` and the `Kernel` context. |
| **Interpreter** | Parses and executes the `.tessera` test suite script. |

---

## ⚠️ Notes on Stability

*   **Version 4.5** fixes critical bugs regarding variable scoping in `exec()` calls, indentation handling in the DSL, and robust dependency checking.
*   The **Autoformalizer** now correctly shares the Global Context (`initial_context`) with the interpreter, allowing "Let" declarations in one test to be visible in the environment.

## 📄 License
Open Source / MIT License.
*Developed for the LNL-Tessera Research Project.*
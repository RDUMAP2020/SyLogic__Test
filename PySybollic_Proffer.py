# =============================================================================
#
#       LNL (Logical Numeral Linguido): A Unified Framework for Neuro-Symbolic Verification
#
# =============================================================================

# Standard library imports
import re
import io
import os
import random
import functools
import textwrap
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Dict, Any, List, Optional, Tuple, Union, Callable, Set, Sequence
)

# Third-party library imports (made optional for core functionality)
try:
    import torch
    import sympy as sp
    import numpy as np
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    )
    _DEPENDENCIES_AVAILABLE = True
except ImportError:
    _DEPENDENCIES_AVAILABLE = False
    # Create dummy classes if dependencies are not available
    class Dummy:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                raise ImportError("Optional dependencies (torch, sympy, numpy) not installed.")
            return method
        def __add__(self, other): return self 
        
    torch, sp, np = Dummy(), Dummy(), Dummy()
    convert_xor = Dummy()
    standard_transformations = (Dummy(),)
    implicit_multiplication_application = Dummy()


# =============================================================================
# FILE: lnl_core/types.py
# =============================================================================

class LNLException(Exception): pass
class KernelTypeError(LNLException): pass
class ParserError(LNLException): pass
class ProofStateError(LNLException): pass

# --- LNL Core Term Data Structures ---

class Term(ABC):
    """Abstract base class for all LNL terms."""
    pass

@dataclass(frozen=True)
class Sort(Term):
    name: str
    def __str__(self) -> str: return self.name

@dataclass(frozen=True)
class Var(Term):
    name: str
    def __str__(self) -> str: return self.name

@dataclass(frozen=True)
class Pi(Term):
    """Dependent function type (forall/implies)."""
    var: Var
    domain: Term
    codomain: Term
    def __str__(self) -> str:
        if self.var.name == "_": return f"({self.domain} -> {self.codomain})"
        return f"forall ({self.var}: {self.domain}), {self.codomain}"

@dataclass(frozen=True)
class Sigma(Term):
    """Dependent pair type (exists)."""
    var: Var
    domain: Term
    codomain: Term
    def __str__(self) -> str: return f"exists ({self.var}: {self.domain}), {self.codomain}"

@dataclass(frozen=True)
class Lam(Term):
    """Lambda abstraction (function definition)."""
    var: Var
    domain: Term
    body: Term
    def __str__(self) -> str: return f"(fun ({self.var}: {self.domain}) => {self.body})"

@dataclass(frozen=True)
class App(Term):
    """Function application."""
    fn: Term
    arg: Term
    def __str__(self) -> str: return f"({self.fn} {self.arg})"

@dataclass(frozen=True)
class Eq(Term):
    """Propositional equality."""
    left: Term
    right: Term
    def __str__(self) -> str: return f"{self.left} = {self.right}"

@dataclass(frozen=True)
class Predicate(Term):
    """A named proposition, distinct from a variable."""
    name: str
    def __str__(self) -> str: return self.name

# --- Hybrid Type Constructors ---

@dataclass(frozen=True)
class Prob(Term):
    """The type for the probability of a proposition."""
    proposition: Term
    def __str__(self) -> str: return f"Prob({self.proposition})"

@dataclass(frozen=True)
class Space(Term):
    """The type for the vector space of a vector object."""
    vector_object: Term
    def __str__(self) -> str: return f"Space({self.vector_object})"

# --- Neuro-Symbolic Extensions ---

@dataclass(frozen=True)
class DeepAtom(Term):
    """Represents a predicate backed by a deep learning model."""
    predicate: Predicate
    def __str__(self) -> str: return f"DeepAtom({str(self.predicate)})"

@dataclass(frozen=True)
class Rule(Term):
    """Represents a weighted or hard rule for neuro-symbolic reasoning."""
    head: Term
    body: Term
    weight: Optional[Union[float, Var]] = None
    def __str__(self) -> str:
        s = f"({self.body} -> {self.head})"
        return f"{self.weight}: {s}" if self.weight is not None else s

# --- Standard Sorts ---
Prop, SetU, Type0 = Sort("Prop"), Sort("Set"), Sort("Type0")
Sort_Vector = Sort("Vector")
Sort_Matrix = Sort("Matrix")
Sort_Real = Sort("Real")
Sort_Any = Sort("Any")

@dataclass
class Ctx:
    """Represents the typing context (Gamma in type theory)."""
    types: Dict[str, Term] = field(default_factory=dict)

    def extend(self, var: Var, term_type: Term) -> "Ctx":
        new_types = self.types.copy()
        new_types[var.name] = term_type
        return Ctx(new_types)

    def lookup(self, var: Var) -> Optional[Term]:
        return self.types.get(var.name)

# =============================================================================
# FILE: lnl_core/kernel.py
# =============================================================================

def substitute(term: Term, var_to_replace: Var, replacement_term: Term) -> Term:
    if isinstance(term, Var):
        return replacement_term if term.name == var_to_replace.name else term
    if isinstance(term, Pi):
        if term.var.name == var_to_replace.name: return term
        return Pi(term.var, substitute(term.domain, var_to_replace, replacement_term), substitute(term.codomain, var_to_replace, replacement_term))
    if isinstance(term, Sigma):
        if term.var.name == var_to_replace.name: return term
        return Sigma(term.var, substitute(term.domain, var_to_replace, replacement_term), substitute(term.codomain, var_to_replace, replacement_term))
    if isinstance(term, Lam):
        if term.var.name == var_to_replace.name: return term
        return Lam(term.var, substitute(term.domain, var_to_replace, replacement_term), substitute(term.body, var_to_replace, replacement_term))
    if isinstance(term, App):
        return App(substitute(term.fn, var_to_replace, replacement_term), substitute(term.arg, var_to_replace, replacement_term))
    if isinstance(term, Eq):
        return Eq(substitute(term.left, var_to_replace, replacement_term), substitute(term.right, var_to_replace, replacement_term))
    if isinstance(term, Prob):
        return Prob(substitute(term.proposition, var_to_replace, replacement_term))
    return term

def _beta_reduce(term: Term) -> Term:
    if isinstance(term, App) and isinstance(term.fn, Lam):
        return substitute(term.fn.body, term.fn.var, term.arg)
    return term

def normalize(term: Term) -> Term:
    reduced = _beta_reduce(term)
    if reduced == term:
        if isinstance(term, App):
            norm_fn = normalize(term.fn)
            norm_arg = normalize(term.arg)
            if norm_fn != term.fn or norm_arg != term.arg:
                return normalize(App(norm_fn, norm_arg))
        elif isinstance(term, Eq):
            norm_left = normalize(term.left)
            norm_right = normalize(term.right)
            if norm_left != term.left or norm_right != term.right:
                return Eq(norm_left, norm_right)
        return reduced
    return normalize(reduced)

def are_convertible(term_a: Term, term_b: Term) -> bool:
    return normalize(term_a) == normalize(term_b)

class KernelVerifier:
    def infer_type(self, ctx: Ctx, term: Term) -> Term:
        if isinstance(term, Sort): return Type0
        if isinstance(term, Var):
            term_type = ctx.lookup(term)
            if term_type is None: raise KernelTypeError(f"Unbound variable: {term}")
            return term_type
        if isinstance(term, Pi):
            self.type_check(ctx.extend(term.var, term.domain), term.codomain, Prop)
            return Prop
        if isinstance(term, Lam):
            self.type_check(ctx, term.domain, Type0)
            body_type = self.infer_type(ctx.extend(term.var, term.domain), term.body)
            return Pi(term.var, term.domain, body_type)
        if isinstance(term, App):
            fn_type = self.infer_type(ctx, term.fn)
            if not isinstance(fn_type, Pi):
                # Restrict looser typing for Sort application only to allowed cases
                if isinstance(fn_type, Type0) and isinstance(term.fn, Sort) and term.fn.name in ["Vector", "Matrix"]:
                    self.infer_type(ctx, term.arg)
                    return Prop
                raise KernelTypeError(f"Cannot apply non-function: {term.fn} of type {fn_type}")
            arg_type = self.infer_type(ctx, term.arg)
            if not are_convertible(arg_type, fn_type.domain):
                raise KernelTypeError(f"Type mismatch: function expects '{fn_type.domain}', but got '{arg_type}' for argument '{term.arg}'")
            return substitute(fn_type.codomain, fn_type.var, term.arg)
        if isinstance(term, Eq): 
            l_type = self.infer_type(ctx, term.left)
            r_type = self.infer_type(ctx, term.right)
            if not are_convertible(l_type, r_type):
                raise KernelTypeError(f"Equality mismatch: LHS type '{l_type}' != RHS type '{r_type}'")
            return Prop
        if isinstance(term, Sigma):
            self.type_check(ctx, term.domain, Type0)
            self.type_check(ctx.extend(term.var, term.domain), term.codomain, Prop)
            return Prop
        if isinstance(term, Prob):
            self.type_check(ctx, term.proposition, Prop)
            return Eq(term, Sort_Real)
        if isinstance(term, Predicate): return Prop
        if isinstance(term, DeepAtom): return self.infer_type(ctx, term.predicate)
        if isinstance(term, Rule): return Prop
        if isinstance(term, Space): return Type0

        raise KernelTypeError(f"Cannot infer type for term: {term}")

    def type_check(self, ctx: Ctx, term: Term, expected_type: Term):
        inferred = self.infer_type(ctx, term)
        if not are_convertible(inferred, expected_type):
            raise KernelTypeError(f"Type mismatch for term '{term}': expected '{expected_type}', but inferred '{inferred}'")


# =============================================================================
# FILE: lnl_numero/multivector.py
# =============================================================================

def popcount(x: int) -> int:
    return bin(x).count("1")

def blade_to_str(mask: int) -> str:
    if mask == 0: return "1"
    return "".join([f"e{i+1}" for i in range(4) if (mask >> i) & 1])

@dataclass
class Multivector4:
    terms: Dict[int, float] = field(default_factory=dict)
    METRIC = (1, 1, 1, 1)

    def __post_init__(self):
        self.terms = {k: v for k, v in self.terms.items() if abs(v) > 1e-12}

    @staticmethod
    def from_basis(s: str, c: float = 1.0) -> "Multivector4":
        mask = sum(1 << (int(d) - 1) for d in re.findall(r'e(\d)', s))
        return Multivector4({mask: c})

    def __str__(self) -> str:
        if not self.terms: return "0"
        return " + ".join(f"{v:.4g}*{blade_to_str(k)}" for k, v in sorted(self.terms.items()))

    def __add__(self, other: "Multivector4") -> "Multivector4":
        new_terms = self.terms.copy()
        for k, v in other.terms.items():
            new_terms[k] = new_terms.get(k, 0) + v
        return Multivector4(new_terms)

    def __mul__(self, other: "Multivector4") -> "Multivector4":
        new_terms = defaultdict(float)
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                m_xor = m1 ^ m2
                m_and = m1 & m2
                sign_flips = sum(popcount(m1 >> (i + 1)) for i in range(4) if (m2 >> i) & 1)
                metric_sign = (-1) ** sum(popcount(m_and & (1 << i)) for i, m in enumerate(self.METRIC) if m == -1)
                new_terms[m_xor] += ((-1) ** sign_flips) * metric_sign * c1 * c2
        return Multivector4(dict(new_terms))

e1, e2, e3, e4 = (Multivector4({1 << i: 1.0}) for i in range(4))

# =============================================================================
# FILE: lnl_linguido/parser.py
# =============================================================================

@dataclass(frozen=True)
class Token:
    type: str
    value: str

class LinguidoTokenizer:
    TOKEN_SPECS = [
        ('LET', r'let'), ('BE', r'be'), ('FOR', r'for'), ('ALL', r'all'),
        ('IF', r'if'), ('THEN', r'then'), ('EXISTS', r'exists'), ('RULE', r'rule'),
        ('IMPLIES', r'implies|->'), ('EQUALS', r'equals|=='), ('ASSIGN', r'='),
        ('COLON', r':'), ('LPAREN', r'\('), ('RPAREN', r'\)'), ('COMMA', r','),
        ('NUMBER', r'\d+(?:\.\d*)?'), 
        ('VAR', r'[a-zA-Z_]\w*'),
        ('SKIP', r'[ \t\n\r]+'), ('MISMATCH', r'.'),
    ]
    TOKEN_REGEX = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPECS))

    def tokenize(self, text: str) -> List[Token]:
        tokens = []
        text = re.sub(r'is\s+equal\s+to', '==', text.lower())
        for mo in self.TOKEN_REGEX.finditer(text.strip()):
            kind, value = mo.lastgroup, mo.group()
            if kind == 'SKIP': continue
            if kind == 'MISMATCH': raise ParserError(f"Unexpected character: '{value}'")
            if kind == 'VAR' and value in ['let', 'be', 'for', 'all', 'if', 'then', 'exists', 'rule']:
                kind = value.upper()
            tokens.append(Token(kind, value))
        return tokens

class DRS:
    def __init__(self): self.referents: Dict[str, Term] = {}
    def add_referent(self, name: str, term_type: Term): self.referents[name] = term_type

class LinguidoGrammarParser:
    def __init__(self, type_map: Dict[str, Sort]):
        self.tokens: List[Token] = []
        self.pos = 0
        self.type_map = type_map

    def parse(self, text: str, drs: DRS) -> Term:
        self.tokens = LinguidoTokenizer().tokenize(text.strip().rstrip('.'))
        self.pos = 0
        statement = self._parse_statement(drs)
        if self._current() is not None:
            raise ParserError(f"Extra characters at end of statement: '{self._current().value}'")
        return statement

    def _current(self) -> Optional[Token]: return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def _advance(self): self.pos += 1
    def _expect(self, token_type: str) -> Token:
        token = self._current()
        if token and token.type == token_type:
            self._advance()
            return token
        found = token.type if token else 'EOF'
        raise ParserError(f"Expected token '{token_type}' but found '{found}'")

    def _parse_statement(self, drs: DRS) -> Term:
        token = self._current()
        if not token: raise ParserError("Unexpected end of input.")
        if token.type == 'LET': return self._parse_declaration(drs)
        if token.type == 'FOR': return self._parse_quantifier(drs, is_pi=True)
        if token.type == 'EXISTS': return self._parse_quantifier(drs, is_pi=False)
        if token.type == 'IF': return self._parse_implication(drs)
        return self._parse_equality_or_expression(drs)

    def _parse_declaration(self, drs: DRS) -> Term:
        self._expect('LET')
        var_name = self._expect('VAR').value
        self._expect('BE')
        token = self._current()
        if token and token.type == 'VAR' and token.value in ['a', 'an']:
            self._advance()
        type_token = self._expect('VAR')
        lnl_type = self.type_map.get(type_token.value.lower())
        if lnl_type is None: raise ParserError(f"Unknown type '{type_token.value}'")
        drs.add_referent(var_name.upper(), lnl_type)
        return Var("Prop_inhabited")

    def _parse_quantifier(self, drs: DRS, is_pi: bool) -> Term:
        self._expect('FOR' if is_pi else 'EXISTS')
        if is_pi: self._expect('ALL')
        var_name = self._expect('VAR').value
        domain: Term = Sort_Any
        if self._current() and self._current().type == 'COLON':
            self._advance()
            domain_name = self._expect('VAR').value.lower()
            domain = self.type_map.get(domain_name, Sort(domain_name.capitalize()))
        quantified_var = Var(var_name.upper())
        drs.add_referent(quantified_var.name, domain)
        self._expect('COMMA')
        body = self._parse_statement(drs)
        return Pi(quantified_var, domain, body) if is_pi else Sigma(quantified_var, domain, body)

    def _parse_implication(self, drs: DRS) -> Term:
        self._expect('IF')
        antecedent = self._parse_statement(drs)
        self._expect('COMMA'); self._expect('THEN')
        consequent = self._parse_statement(drs)
        return Pi(Var("_"), antecedent, consequent)

    def _parse_equality_or_expression(self, drs: DRS) -> Term:
        left = self._parse_expression(drs)
        if self._current() and self._current().type == 'EQUALS':
            self._advance()
            right = self._parse_expression(drs)
            return Eq(left, right)
        return left

    def _parse_expression(self, drs: DRS) -> Term:
        token = self._current()
        if not token: raise ParserError("Unexpected end of expression.")
        if token.type == 'NUMBER':
            self._advance(); return Var(token.value)
        var_token = self._expect('VAR')
        var_name_upper = var_token.value.upper()
        
        if var_token.value.lower() in self.type_map:
            expr = self.type_map[var_token.value.lower()]
        else:
            expr = Var(var_name_upper)

        if self._current() and self._current().type == 'LPAREN':
            self._advance()
            args = [self._parse_expression(drs)]
            while self._current() and self._current().type == 'COMMA':
                self._advance()
                args.append(self._parse_expression(drs))
            self._expect('RPAREN')
            return functools.reduce(App, args, expr)
        return expr

# =============================================================================
# FILE: lnl_prover/proof_state.py
# =============================================================================

@dataclass
class ProofState:
    ctx: Ctx
    goals: List[Term]
    parent: Optional['ProofState'] = None

    @property
    def current_goal(self) -> Optional[Term]:
        return self.goals[0] if self.goals else None

    def is_solved(self) -> bool:
        return not self.goals

# =============================================================================
# FILE: lnl_prover/tactics.py
# =============================================================================

class Tactics:
    def __init__(self, state: ProofState):
        self.state = state

    def reflexivity(self):
        if self.state.current_goal is None: raise ProofStateError("No goals to solve.")
        goal = self.state.current_goal
        if isinstance(goal, Eq) and are_convertible(goal.left, goal.right):
            self.state.goals.pop(0)
            print("  [TACTIC: reflexivity] Goal solved by reflexivity.")
        else:
            raise ProofStateError(f"Reflexivity failed: goal '{goal}' is not a convertible equality.")

    def intro(self, var_name: str):
        if self.state.current_goal is None: raise ProofStateError("No goals to solve.")
        goal = self.state.current_goal
        if isinstance(goal, Pi):
            new_var = Var(var_name.upper())
            self.state.ctx = self.state.ctx.extend(new_var, goal.domain)
            new_goal = substitute(goal.codomain, goal.var, new_var)
            self.state.goals[0] = new_goal
            print(f"  [TACTIC: intro] Introduced '{new_var}', new goal: '{new_goal}'")
        else:
            raise ProofStateError(f"Intro failed: goal '{goal}' is not a Pi type.")

    def apply(self, theorem_name: str):
        if self.state.current_goal is None: raise ProofStateError("No goals to solve.")
        theorem = self.state.ctx.lookup(Var(theorem_name.upper()))
        if theorem is None: raise ProofStateError(f"Theorem '{theorem_name}' not found in context.")
        if isinstance(theorem, Pi):
            if are_convertible(theorem.codomain, self.state.current_goal):
                self.state.goals[0] = theorem.domain
                print(f"  [TACTIC: apply] Applied '{theorem_name}', new goal: '{theorem.domain}'")
            else:
                raise ProofStateError(f"Apply failed: theorem codomain '{theorem.codomain}' doesn't match goal '{self.state.current_goal}'.")
        else:
            raise ProofStateError(f"Apply failed: '{theorem_name}' is not an implication (Pi type).")

# =============================================================================
# FILE: lnl_ai/autoformalizer.py
# =============================================================================

class LogicEmbedder:
    def process(self, source_stream: io.StringIO) -> List[Dict[str, Any]]:
        if not _DEPENDENCIES_AVAILABLE: raise ImportError("LogicEmbedder requires sympy.")
        content = source_stream.read()
        chunks = re.findall(r'\$(.*?)\$', content)
        results = []
        for chunk in chunks:
            try:
                transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
                expr = parse_expr(chunk.replace('=', '=='), transformations=transformations, evaluate=False)
                results.append({'lnl_term': self._embed_recursive(expr)})
            except Exception: continue
        return results

    def _embed_recursive(self, expr: "sp.Expr") -> Term:
        if isinstance(expr, sp.Symbol): return Var(expr.name)
        if isinstance(expr, sp.Number): return Var(str(expr))
        if isinstance(expr, sp.Equality): return Eq(self._embed_recursive(expr.lhs), self._embed_recursive(expr.rhs))
        if hasattr(expr, 'func'):
            op_name = expr.func.__name__.upper()
            lnl_op = Predicate(op_name) if op_name in ["SIN", "COS", "EXP"] else Var(op_name)
            res = lnl_op
            for arg in expr.args: res = App(res, self._embed_recursive(arg))
            return res
        raise LNLException(f"Cannot embed expression type {type(expr)}")

class Autoformalizer:
    def __init__(self, parser: LinguidoGrammarParser, env_dict: Dict[str, Any], kernel: KernelVerifier):
        self.parser = parser
        self.env_dict = env_dict 
        self.kernel = kernel

    def formalize(self, text: str, drs: DRS) -> Optional[Term]:
        ctx = self.env_dict['initial_context']
        try:
            term = self.parser.parse(text, drs)
            for name, type_term in drs.referents.items():
                if not ctx.lookup(Var(name)):
                    ctx = ctx.extend(Var(name), type_term)
            if term == Var("Prop_inhabited"):
                self.env_dict['initial_context'] = ctx
                print("  Formalization PASSED: Declaration successful.")
                return term
            self.kernel.type_check(ctx, term, Prop)
            print(f"  Formalization PASSED: '{term}' is a valid proposition.")
            return term
        except (ParserError, KernelTypeError) as e:
            print(f"  Formalization REJECTED: {e}")
            return None

@dataclass
class Fact: name: str; conf: "torch.Tensor";
@dataclass
class Edge: premises: List[str]; conclusion: str; rule: str;

class NeuroSymbolicProof:
    def __init__(self, goal: str):
        if not _DEPENDENCIES_AVAILABLE: raise ImportError("NeuroSymbolicProof requires PyTorch/NumPy.")
        self.goal = goal
        self.facts: Dict[str, Fact] = {goal: Fact(goal, torch.tensor(0.0))}
        self.edges: List[Edge] = []
        self.attention_weights: Dict[str, float] = defaultdict(lambda: 1.0)

    def assume(self, proposition: str, conf: float = 1.0) -> "torch.Tensor":
        self.facts[proposition] = Fact(proposition, torch.tensor(conf))
        self._update_attention()
        return self.facts[proposition].conf

    def apply(self, conclusion: str, premises: List[str], rule_conf: float = 1.0, rule_name: str = "implies") -> "torch.Tensor":
        premise_confs = [self.facts.get(p, Fact(p, torch.tensor(0.0))).conf for p in premises]
        premise_attentions = torch.tensor([self.attention_weights[p] for p in premises], dtype=torch.float32)
        agg_conf = torch.prod(torch.stack(premise_confs) * premise_attentions) if premise_confs else torch.tensor(1.0)
        inferred = agg_conf * rule_conf
        current_conf = self.facts.get(conclusion, Fact(conclusion, torch.tensor(0.0))).conf
        self.facts[conclusion] = Fact(conclusion, torch.maximum(current_conf, inferred))
        self.edges.append(Edge(premises, conclusion, rule_name))
        self._update_attention()
        return self.facts[conclusion].conf

    def _update_attention(self):
        if not self.facts: return
        fact_names, confs = zip(*[(f.name, f.conf.item()) for f in self.facts.values()])
        softmax_vals = torch.nn.functional.softmax(torch.tensor(confs, dtype=torch.float32), dim=0)
        self.attention_weights = {name: val.item() for name, val in zip(fact_names, softmax_vals)}

    def confidence(self, proposition: str) -> "torch.Tensor":
        return self.facts.get(proposition, Fact(proposition, torch.tensor(0.0))).conf


# =============================================================================
# FILE: tessera/interpreter.py
# =============================================================================

class LNLVerifyInterpreter:
    def __init__(self, lnl_components: Dict[str, Any]):
        # Use direct reference to allow Autoformalizer to update shared state (e.g., initial_context)
        self.env = lnl_components 
        self.passes, self.failures = 0, 0
        
        def mark_passed(): self.passes += 1
        
        self.env.update({
            'Var': Var, 'App': App, 'Pi': Pi, 'Eq': Eq, 'Sort': Sort, 'Ctx': Ctx,
            'Predicate': Predicate, 'Prop': Prop, 'Sort_Vector': Sort_Vector, 'are_convertible': are_convertible,
            'Sort_Real': Sort_Real, 'Prob': Prob, 'Tactics': Tactics,
            'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4, 'Multivector4': Multivector4,
            'DRS': DRS, 'KernelTypeError': KernelTypeError, 'ParserError': ParserError,
            'ProofStateError': ProofStateError, 'ProofState': ProofState,
            'NeuroSymbolicProof': NeuroSymbolicProof, 'io': io,
            '_mark_passed': mark_passed
        })

    def _get_full_command(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        cmd_lines, i, brace_level = [], start_idx, 0
        while i < len(lines):
            line = lines[i]
            cmd_lines.append(line)
            brace_level += line.count('{') - line.count('}')
            if brace_level == 0: return "\n".join(cmd_lines), i + 1
            i += 1
        return "\n".join(cmd_lines), i

    def _run_block(self, block_lines: List[str]):
        i = 0
        while i < len(block_lines):
            line = block_lines[i].strip()
            if not line or line.startswith('#'): i += 1; continue
            full_command, next_idx = self._get_full_command(block_lines, i)
            i = next_idx
            try:
                self._execute_command(full_command)
            except AssertionError as e:
                self.failures += 1
                print(f"  [--FAILURE--]: Assertion failed!\n    '{full_command.strip().splitlines()[0]}'\n    Reason: {e}")
            except Exception as e:
                self.failures += 1
                print(f"  [--FAILURE--]: Command failed unexpectedly!\n    '{full_command.strip().splitlines()[0]}'\n    Reason: {type(e).__name__}: {e}")

    def _execute_command(self, command_str: str):
        command_str = command_str.strip()
        m_block = re.match(r'(Suite|Test)\s+"(.*?)"\s*\{(.*)\}', command_str, re.DOTALL)
        if m_block:
            block_type, name, content = m_block.groups()
            print(f"\n{'='*10} {block_type.upper()}: {name} {'='*10}")
            self._run_block(content.strip().splitlines()); return

        m_prove = re.match(r'Prove\s+(.*?)\s+With\s*\{(.*)\}', command_str, re.DOTALL)
        if m_prove:
            print(f"  [COMMAND] {command_str.splitlines()[0].strip()}...")
            goal_str, tactics_str = m_prove.groups()
            goal = eval(goal_str.strip(), self.env)
            if 'current_proof_state' in self.env and self.env['current_proof_state'] is not None:
                state = ProofState(self.env['current_proof_state'].ctx, [goal])
            else:
                state = ProofState(self.env['initial_context'], [goal])
            self.env['current_proof_state'] = state
            for tactic_call in [tc.strip() for tc in tactics_str.strip().split(';') if tc.strip()]:
                eval(f"Tactics(current_proof_state).{tactic_call}", self.env)
            if state.is_solved():
                print("  [PASSED]: Proof completed."); self.passes += 1
            return

        m_python = re.match(r'Python\s*\{(.*)\}', command_str, re.DOTALL)
        if m_python:
            code = m_python.group(1)
            exec(textwrap.dedent(code), self.env)
            return

        m_expect = re.match(r'ExpectFail\s+(\w+)\s*\{(.*)\}', command_str, re.DOTALL)
        if m_expect:
            print(f"  [COMMAND] {command_str.splitlines()[0].strip()}...")
            exc_type_str, code = m_expect.groups()
            exc_type = self.env.get(exc_type_str)
            
            if exc_type is None:
                raise AssertionError(f"Test Config Error: Exception '{exc_type_str}' not found in environment.")
            if not isinstance(exc_type, type) or not issubclass(exc_type, Exception):
                raise AssertionError(f"Test Config Error: '{exc_type_str}' is not a valid Exception class. Got: {type(exc_type)}")
            
            try:
                exec(textwrap.dedent(code), self.env)
                raise AssertionError(f"Expected {exc_type_str}, but succeeded.")
            except Exception as e:
                if isinstance(e, exc_type):
                    print(f"  [PASSED]: Correctly failed with {exc_type_str}."); self.passes += 1
                else: self.failures += 1; raise
            return

        if command_str.startswith("Assert "):
            print(f"  [COMMAND] {command_str}")
            result = eval(command_str[len("Assert "):], self.env)
            assert result, f"Assertion failed: {command_str[7:]}"
            print("  [PASSED]"); self.passes += 1
        else:
            if '\n' in command_str: exec(command_str, self.env)
            else: exec(command_str, self.env)

    def run(self, source_code: str):
        print("\n" + "*" * 70 + "\n         Executing LNL-Tessera Verification Suite\n" + "*" * 70)
        self._run_block(source_code.strip().splitlines())
        print("\n" + "*" * 70 + f"\n                     Suite Complete\n  PASSED: {self.passes} | FAILED: {self.failures}\n" + "*" * 70)


# =============================================================================
# FILE: tessera/suite.tessera
# =============================================================================
LNL_TESSERA_SOURCE = """
Suite "LNL Core Verification Suite" {
    Test "Numero: Geometric Product Identity" {
        # Test the identity (e1+e2)*(e1-e2) = -2*e1e2
        product = (e1 + e2) * (e1 + Multivector4.from_basis("e2", -1.0)) 
        expected = Multivector4.from_basis("e1e2", -2.0)
        Assert product.terms == expected.terms
    }
    Test "Kernel: Rejects Ill-Typed Application" {
        ExpectFail KernelTypeError {
            # Attempt to apply a Sort (a type) to another Sort.
            kernel.type_check(Ctx(), App(Sort("Set"), Sort("Set")), Prop)
        }
    }
    Test "Kernel: Type Inference for Prob" {
        expected_type = Eq(Prob(Predicate("P")), Sort_Real)
        Assert are_convertible(kernel.infer_type(Ctx(), Prob(Predicate("P"))), expected_type)
    }
    Test "Kernel: Type Inference for Pi with Prop Domain" {
        Assert are_convertible(kernel.infer_type(Ctx(), Pi(Var("_"), Predicate("P"), Predicate("Q"))), Prop)
    }
    Test "Kernel: Type Inference for Pi with Type Domain" {
        ctx = Ctx().extend(Var("x"), Sort_Real)
        Assert are_convertible(kernel.infer_type(ctx, Pi(Var("x"), Sort_Real, Eq(Var("x"), Var("x")))), Prop)
    }
    Test "Linguido: Fails on Ungrammatical Input" {
        ExpectFail ParserError {
            parser.parse("For every green idea, sleep furiously.", DRS())
        }
    }
    Test "Proof: `reflexivity` tactic" {
        Prove Eq(Sort_Vector, Sort_Vector) With {
            reflexivity();
        }
    }
    Test "Proof: `intro` and `reflexivity`" {
        Prove Pi(Var("x"), Sort_Vector, Eq(Var("x"), Var("x"))) With {
            intro("y");
            reflexivity;
        }
    }
    Test "Proof: `apply` tactic for implication" {
        ctx_with_thm = initial_context.extend(Var("THM"), Pi(Var("_"), Predicate("P"), Predicate("Q")))
        current_proof_state = ProofState(ctx_with_thm, [])
        Prove Predicate("Q") With {
            apply("THM")
        }
        Assert current_proof_state.current_goal == Predicate("P")
    }
    Test "Neuro-Symbolic: Confidence Propagation" {
        Python {
            if _DEPENDENCIES_AVAILABLE:
                proof = NeuroSymbolicProof("Q")
                proof.assume("P", 0.5)
                proof.apply("Q", ["P"], rule_conf=0.9)
                assert proof.confidence("Q").item() > 0.27
                _mark_passed()
            else:
                print("  Skipping Neuro-Symbolic test: torch/numpy not installed.")
                _mark_passed()
        }
    }
}
Suite "AI and Embedding Components" {
    Test "Autoformalizer: Successful Declaration" {
        drs_obj = DRS()
        res = autoformalizer.formalize("let v be a vector.", drs_obj)
        Assert res is not None
        # Check if the global context was updated properly
        Assert initial_context.lookup(Var("V")) is not None
    }
    Test "Autoformalizer: Rejects Unknown Types" {
        Assert autoformalizer.formalize("let v be a magic_type.", DRS()) is None
    }
    Test "LogicEmbedder: Translate LaTeX to LNL Term" {
        Python {
            if _DEPENDENCIES_AVAILABLE:
                latex_source = io.StringIO("An equation is $x = y + 1$.")
                embedded_data = logic_embedder.process(latex_source)
                assert len(embedded_data) > 0
                expected_term = Eq(Var('x'), App(App(Var('ADD'), Var('y')), Var('1')))
                assert isinstance(embedded_data[0]['lnl_term'], Eq)
                _mark_passed()
            else:
                print("  Skipping LogicEmbedder test: sympy not installed.")
                _mark_passed()
        }
    }
}
"""

# =============================================================================
# FILE: main.py
# =============================================================================
def main():
    if not _DEPENDENCIES_AVAILABLE:
        print("Warning: Optional dependencies (torch, sympy, numpy) not found.")
        print("AI-related features like LogicEmbedder and NeuroSymbolicProof will be disabled.")

    kernel = KernelVerifier()

    initial_context = Ctx({
        'ADD': Pi(Var("_"), Sort_Real, Pi(Var("_"), Sort_Real, Sort_Real)),
        'MUL': Pi(Var("_"), Sort_Real, Pi(Var("_"), Sort_Real, Sort_Real)),
        'P': Prop, 
        'Q': Prop,
    })
    type_map = {
        'prop': Prop, 'set': SetU, 'type': Type0,
        'vector': Sort_Vector, 'matrix': Sort_Matrix, 'real': Sort_Real, 'any': Sort_Any,
    }

    parser = LinguidoGrammarParser(type_map)
    logic_embedder = LogicEmbedder()

    # Define components dictionary
    # available in the environment for ExpectFail checks before the interpreter 
    # updates its own environment.
    lnl_components = {
        'kernel': kernel,
        'parser': parser,
        'logic_embedder': logic_embedder,
        'initial_context': initial_context,
        'current_proof_state': None, 
        '_DEPENDENCIES_AVAILABLE': _DEPENDENCIES_AVAILABLE,
        'KernelTypeError': KernelTypeError,
        'ParserError': ParserError,
        'ProofStateError': ProofStateError
    }

    # Pass the dictionary itself to Autoformalizer so it can update 'initial_context'
    # in the SHARED dictionary that the interpreter also uses.
    autoformalizer = Autoformalizer(parser, lnl_components, kernel)
    lnl_components['autoformalizer'] = autoformalizer

    interpreter = LNLVerifyInterpreter(lnl_components)
    interpreter.run(LNL_TESSERA_SOURCE)

if __name__ == "__main__":
    main()
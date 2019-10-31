
use failure::{Error};
use std::rc::{Rc};
use std::convert::TryInto;
use std::ops::{Range};
use derivative::Derivative;

#[derive(Debug, PartialEq)]
struct Baresymbol(String);

fn baresymbol(name: &str) -> Baresymbol {
    Baresymbol(String::from(name))
}

#[derive(Debug, PartialEq)] // do not add Clone !
// ^ XX really add PartialEq? Really mis-use Globalsymbol as symbol in Value?
struct Globalsymbol {
    namesym: Baresymbol,
    value: Option<Value>, // XX mut
}

fn globalsymbol(name: &str) -> Rc<Globalsymbol> {
    Rc::new(Globalsymbol { namesym: baresymbol(name), value: None })
}

fn globalsymbol_bound(name: &str, v: Value) -> Rc<Globalsymbol> {
    Rc::new(Globalsymbol { namesym: baresymbol(name), value: Some(v) })
}


#[derive(Debug, PartialEq)]
struct RawPair (Value, Value);

#[derive(Debug, PartialEq)]
struct Body {
    nvars: u16,
    // ^ the number of free local variables in the body
    expr: Rc<Expr>,
}


// Hack: "typedef pattern" and casting to make PartialEq work (on the
// current Rust version)
#[derive(Clone)]
struct Primitive2proc(fn (Value, Value) -> Value);
impl PartialEq for Primitive2proc {
    fn eq(&self, other: &Primitive2proc) -> bool {
        match (self, other) {
            (&Primitive2proc(a), &Primitive2proc(b))
                =>  a as usize == b as usize
        }
    }
}
#[derive(Clone)]
struct PrimitiveNproc(fn (&Vec<Value>) -> Value);
impl PartialEq for PrimitiveNproc {
    fn eq(&self, other: &PrimitiveNproc) -> bool {
        match (self, other) {
            (&PrimitiveNproc(a), &PrimitiveNproc(b))
                =>  a as usize == b as usize
        }
    }
}


#[derive(Clone, PartialEq)]
// PartialEq just because we support first-class functions
#[derive(Derivative)]
#[derivative(Debug)]
enum Callable {
    Function(Rc<Body>),
    Closure {
        env: Vec<Value>,
        proc: Rc<Body>,
    },
    Primitive2 {
        #[derivative(Debug="ignore")]
        proc: Primitive2proc,
    },
    PrimitiveN {
        arity: Range<u16>,
        #[derivative(Debug="ignore")]
        proc: PrimitiveNproc,
    },
}

impl Callable {
    fn apply (&self, argvals: &mut dyn std::iter::Iterator<Item = Value>) -> Value {
        match self {
            Callable::Function(lam) => {
                let env_and_args = argvals.collect::<Vec<Value>>();
                if lam.nvars == env_and_args.len().try_into().unwrap() {
                    // evaluate the lambda's body in this new env
                    (*(lam.expr)).eval(&env_and_args)
                } else {
                    string("WRONG_ARITY")
                }
            },
            Callable::Closure{env, proc} => {
                string("TODO")
            },
            Callable::Primitive2{proc: Primitive2proc(proc)} =>
            match (|| -> Option<Value> {
                let a1= argvals.next()?;
                let a2= argvals.next()?;
                match argvals.next() {
                    None => Some(proc(a1, a2)),
                    Some(_) => Some(string("TOO MANY ARGUMENTS"))
                }
            })() {
                Some(res) => res,
                None => string("NOT ENOUGH ARGUMENTS")
            },
            Callable::PrimitiveN{arity, proc} => {
                string("TODO")
            },
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum Value {
    Boolean(bool),
    Integer64(i64),
    String(Rc<String>), // h ?
    Nil,
    Pair(Rc<RawPair>),
    Symbol(Rc<Globalsymbol>),
    Void,
    Callable(Callable),
}

fn boolean (v: bool) -> Value { Value::Boolean(v) }
fn integer (v: i64) -> Value { Value::Integer64(v) }
fn string (v: &str) -> Value { Value::String(Rc::new(v.to_string())) }
const NIL : Value = Value::Nil;
fn cons (a: Value, b: Value) -> Value { Value::Pair(Rc::new(RawPair(a, b))) }
//...
fn function(nvars: u16, body: Expr) -> Value {
    Value::Callable(Callable::Function(Rc::new(Body { nvars: nvars,
                                                      expr: Rc::new(body) })))
}
fn primitive2(proc: fn (Value, Value) -> Value) -> Value {
    Value::Callable(Callable::Primitive2 {
        proc: Primitive2proc(proc)
    })
}
fn primitive_n(arity: Range<u16>,
               proc: fn (&Vec<Value>) -> Value) -> Value {
    Value::Callable(Callable::PrimitiveN {
        arity: arity,
        proc: PrimitiveNproc(proc)
    })
}


impl Value {
    fn apply(&self, args: &mut dyn std::iter::Iterator<Item = Value>) -> Value {
        match self {
            Value::Callable(c) => c.apply(args),
            _ => Value::String(Rc::new("NOT_A_CALLABLE".to_string()))
        }
    }
}

#[derive(Debug, PartialEq)]
// PartialEq because Expr is used in Callable (via Body)
enum Expr {
    Literal(Value),
    App(Rc<Expr>, Vec<Rc<Expr>>),
    Lamda(Rc<Body>),
    Globalref(Rc<Globalsymbol>),
    Localref(u16),
    Globaldef(Rc<Globalsymbol>, Rc<Expr>), // def-once I mean erlangy?
}

fn literal(v: Value) -> Expr { Expr::Literal(v) }
fn app(f: Expr, args: Vec<Expr>) -> Expr {
    Expr::App(Rc::new(f), args.into_iter().map(|e| Rc::new(e)).collect())
}
fn globalref(v: &Rc<Globalsymbol>) -> Expr {
    Expr::Globalref(v.clone())
}


trait Eval {
    fn eval(&self, env: &Vec<Value>) -> Value;
}

impl Eval for Expr {
    fn eval(&self, env: &Vec<Value>) -> Value {
        match self {
            Expr::Literal(v) => v.clone(),
            Expr::Lamda(l) => Value::Callable(Callable::Function(l.clone())),
            Expr::App(p, args) => {
                let pval = p.eval(env);
                let mut argvals = args.iter().map(|v| (*v).eval(env));
                pval.apply(&mut argvals)
            },
            Expr::Localref(id) => {
                let id2 : usize = (*id).try_into().unwrap(); // XX ' as ' ? safe?
                (env[id2]).clone()
            },
            Expr::Globalref(sym) => {
                match &(*sym).value {
                    Some(v) => v.clone(),
                    None => string("UNBOUND")
                }
            },
            _ => Value::String(Rc::new("NOMATCH".to_string()))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn main() {
        let mut errors= 0;
        let mut t= |prog: Expr, expected: Value| {
            let env= Vec::new();
            let res= prog.eval(&env);
            if res != expected {
                errors += 1;
                println!("{:?} = {:?}, expected: {:?}", prog, res, expected);
            }
        };

        t(literal(boolean(false)), boolean(false));
        t(literal(boolean(true)), boolean(true));
        t(literal(string("Hello")), string("Hello"));
        t(literal(cons(string("hi"), NIL)), cons(string("hi"), NIL));

        let wrong_arity_error= string("WRONG_ARITY");
        let not_enough_args_error= string("NOT ENOUGH ARGUMENTS");
        let too_many_args_error= string("TOO MANY ARGUMENTS");

        let unbound_f= globalsymbol("f");
        let x= globalsymbol_bound("x", integer(42));
        let f0= globalsymbol_bound("f0", function(0, globalref(&x)));
        let f1= globalsymbol_bound("f1", function(1, globalref(&x)));
        t(globalref(&f0),
          function(0, globalref(&x)));
        t(app(globalref(&f0), vec![]),
          integer(42));
        t(app(globalref(&f0), vec![globalref(&f0)]),
          wrong_arity_error);
        t(app(globalref(&f1), vec![globalref(&f0)]),
          integer(42));
        let var_cons= globalsymbol_bound("cons", primitive2(cons));
        t(app(globalref(&var_cons),
              vec![app(globalref(&f1), vec![globalref(&f0)]),
                   literal(integer(41))]),
          // (cons (f1 f0) 41)
          cons(integer(42), integer(41)));
        t(app(globalref(&var_cons),
              vec![app(globalref(&f1), vec![globalref(&f0)])]),
          not_enough_args_error);
        t(app(globalref(&var_cons),
              vec![literal(integer(41)),
                   literal(integer(41)),
                   literal(integer(41))]),
          too_many_args_error);
        assert_eq!(errors, 0);
    }

}


fn main() {
    unimplemented!()
}

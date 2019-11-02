
// Not done yet:
//   - proper error system in the guest language
//   - have Value be the size of 1 machine word so it's cheap to copy

use failure::{Error};
use std::rc::{Rc};
use std::convert::TryInto;
use std::ops::{Range};
use derivative::Derivative;

macro_rules! mdo {
    ($t:ty, $e:block) => ( (|| -> $t { $e })() )
}


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
    nvars: u16, // the number of free local variables in the body
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
// PartialEq because we support first-class functions
#[derive(Derivative)]
#[derivative(Debug)]
enum Callable {
    Function(Rc<Body>),
    Closure {
        env: Vec<Value>,
        body: Rc<Body>,
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
            Callable::Function(body) => {
                let env_and_args = argvals.collect::<Vec<Value>>();

                // evaluate the body in this new env if arity is correct
                if body.nvars == env_and_args.len().try_into().unwrap() {
                    (*(body.expr)).eval(&env_and_args)
                } else {
                    string("WRONG_ARITY")
                }
            },
            Callable::Closure{env, body} => {
                let envi= env.iter().map(|v: &Value| v.clone());
                let env_and_args = envi.chain(argvals).collect::<Vec<Value>>();
                
                // COPY-PASTE from above
                if body.nvars == env_and_args.len().try_into().unwrap() {
                    (*(body.expr)).eval(&env_and_args)
                } else {
                    string("WRONG_ARITY")
                }
            },
            Callable::Primitive2{proc: Primitive2proc(proc)} =>
            match mdo!(Option<Value>, {
                let a1= argvals.next()?;
                let a2= argvals.next()?;
                match argvals.next() {
                    None => Some(proc(a1, a2)),
                    Some(_) => Some(string("TOO MANY ARGUMENTS"))
                }
            }) {
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
fn closure(env: Vec<Value>,
           nvars: u16,
           body: Expr) -> Value {
    Value::Callable(Callable::Closure {
        env: env,
        body: Rc::new(Body { nvars: nvars,
                             expr: Rc::new(body) }),
    })
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
            _ => string("NOT_A_CALLABLE")
        }
    }
}

#[derive(Debug, PartialEq)]
// PartialEq because Expr is used in Callable (via Body)
enum Expr {
    Literal(Value),
    App(Rc<Expr>, Vec<Rc<Expr>>),
    Lambda {
        envslots: Option<Vec<u16>>,
        body: Rc<Body>
    },
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
fn localref(i: u16) -> Expr { Expr::Localref(i) }
fn lambda(envslots: Option<Vec<u16>>,
          arity: u16,
          body: Expr) -> Expr {
    let nvars= match &envslots {
        Some(slots) => slots.len() as u16 + arity, // XX danger cast?
        None => arity
    };
    Expr::Lambda { envslots: envslots,
                   body: Rc::new(Body { nvars: nvars,
                                        expr: Rc::new(body)})}
}


trait Eval {
    fn eval(&self, env: &Vec<Value>) -> Value;
}

impl Eval for Expr {
    fn eval(&self, env: &Vec<Value>) -> Value {
        match self {
            Expr::Literal(v) => v.clone(),
            Expr::Lambda {envslots, body} => {
                match envslots {
                    Some(slots) => {
                        Value::Callable
                            (Callable::Closure
                             {
                                 env: {
                                     slots
                                         .into_iter()
                                         .map(|i| env[*i as usize].clone())
                                         .collect()
                                 },
                                 body: body.clone()
                             })
                    },
                    None => Value::Callable(Callable::Function(body.clone())),
                }
            }
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
            _ => string("NOMATCH")
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

        for v in &[boolean(false),
                   boolean(true),
                   string("Hello"),
                   cons(string("hi"), NIL)] {
            t(literal(v.clone()), v.clone())
        }

        let wrong_arity_error= string("WRONG_ARITY");
        let not_enough_args_error= string("NOT ENOUGH ARGUMENTS");
        let too_many_args_error= string("TOO MANY ARGUMENTS");

        // let unbound_f= globalsymbol("f");
        let _x= globalsymbol_bound("x", integer(42));
        let _f0= globalsymbol_bound("f0", function(0, globalref(&_x)));
        let _f1= globalsymbol_bound("f1", function(1, globalref(&_x)));

        t(globalref(&_f0),
          function(0, globalref(&_x)));
        t(app(globalref(&_f0), vec![]),
          integer(42));
        t(app(globalref(&_f0), vec![globalref(&_f0)]),
          wrong_arity_error);
        t(app(globalref(&_f1), vec![globalref(&_f0)]),
          integer(42));

        let _cons= globalsymbol_bound("cons", primitive2(cons));

        // (cons (f1 f0) 41)
        t(app(globalref(&_cons),
              vec![app(globalref(&_f1), vec![globalref(&_f0)]),
                   literal(integer(41))]),
          cons(integer(42), integer(41)));

        // (cons (f1 f0))
        t(app(globalref(&_cons),
              vec![app(globalref(&_f1), vec![globalref(&_f0)])]),
          not_enough_args_error);

        // (cons 41 41 41)
        t(app(globalref(&_cons),
              vec![literal(integer(41)),
                   literal(integer(41)),
                   literal(integer(41))]),
          too_many_args_error);

        // (lambda () 41)
        let e_simple= lambda(None, 0, literal(integer(41)));
        t(e_simple,
          function(0, literal(integer(41))));
        
        // (lambda (x) (lambda (y) (cons x y)))
        let e_ccons= lambda(None, 1,
                            lambda(Some(vec![0]), 1,
                                   app(globalref(&_cons),
                                       vec![localref(0),
                                            localref(1)])));
        t(app(app(e_ccons,
                  vec![literal(integer(41))]),
              vec![literal(integer(42))]),
          cons(integer(41), integer(42)));

        assert_eq!(errors, 0);
    }

}


fn main() {
    unimplemented!()
}

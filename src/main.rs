
use failure::{Error};
use std::rc::{Rc};
use std::convert::TryInto;
use std::ops::{Range};
use derivative::Derivative;

#[derive(Debug)]
struct Baresymbol(String);

fn baresymbol(name: &str) -> Baresymbol {
    Baresymbol(String::from(name))
}

#[derive(Debug)] // do not add Clone !
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


#[derive(Debug)]
struct RawPair (Value, Value);

#[derive(Debug)]
struct RawLambda {
    // with no captured env
    arityWithEnv: u16,
    // ^ ok?, this is the number of free local variables in the
    // body. (A "low-level arity" i.e. number of *variables* not
    // argument values, (lambda (a b . r) ..) will have an arity of
    // just 3.)
    body: Rc<Expr>,
}


#[derive(Clone)]
#[derive(Derivative)]
#[derivative(Debug)]
enum Callable {
    Function(Rc<RawLambda>),
    Closure {
        env: Vec<Value>,
        proc: Rc<RawLambda>,
    },
    Primitive2 {
        #[derivative(Debug="ignore")]
        proc: fn (Value, Value) -> Value,
    },
    PrimitiveN {
        arity: Range<u16>,
        #[derivative(Debug="ignore")]
        proc: fn (&Vec<Value>) -> Value,
    },
}

impl Callable {
    fn apply (&self, argvals: &mut dyn std::iter::Iterator<Item = Value>) -> Value {
        match self {
            Callable::Function(lam) => {
                let envAndArgs = argvals.collect::<Vec<Value>>();
                if lam.arityWithEnv == envAndArgs.len().try_into().unwrap() {
                    // evaluate the lambda's body in this new env
                    (*(lam.body)).eval(&envAndArgs)
                } else {
                    string("WRONG_ARITY")
                }
            },
            Callable::Closure{env, proc} => {
                string("TODO")
            },
            Callable::Primitive2{proc} =>
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

#[derive(Debug, Clone)]
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
const nil : Value = Value::Nil;
fn cons (a: Value, b: Value) -> Value { Value::Pair(Rc::new(RawPair(a, b))) }
//...
fn function(nvars: u16, body: Expr) -> Value {
    Value::Callable(Callable::Function(Rc::new(RawLambda { arityWithEnv: nvars,
                                                           body: Rc::new(body) })))
}
fn primitive2(proc: fn (Value, Value) -> Value) -> Value {
    Value::Callable(Callable::Primitive2 {
        proc: proc
    })
}
fn primitiveN(arity: Range<u16>,
              proc: fn (&Vec<Value>) -> Value) -> Value {
    Value::Callable(Callable::PrimitiveN {
        arity: arity,
        proc: proc
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

#[derive(Debug)]
enum Expr {
    Literal(Value),
    App(Rc<Expr>, Vec<Rc<Expr>>),
    Lamda(Rc<RawLambda>),
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


fn eval(prog: Expr) {
    let env= Vec::new();
    println!("{:?} = {:?}", prog, prog.eval(&env));
}

fn main() {
    eval(literal(boolean(false)));
    eval(literal(boolean(true)));
    eval(literal(string("Hello")));
    eval(literal(cons(string("hi"), nil)));

    let unbound_f= globalsymbol("f");
    let x= globalsymbol_bound("x", integer(42));
    let f0= globalsymbol_bound("f0", function(0, globalref(&x)));
    let f1= globalsymbol_bound("f1", function(1, globalref(&x)));
    eval(globalref(&f0));
    eval(app(globalref(&f0), vec![]));
    eval(app(globalref(&f0), vec![globalref(&f0)])); // wrong arity
    eval(app(globalref(&f1), vec![globalref(&f0)]));
    let var_cons= globalsymbol_bound("cons", primitive2(cons));
    eval(app(globalref(&var_cons),
             vec![app(globalref(&f1), vec![globalref(&f0)]),
                  literal(integer(41))])); // (cons (f1 f0) 41)
    eval(app(globalref(&var_cons),
             vec![app(globalref(&f1), vec![globalref(&f0)])])); // not enough args
    eval(app(globalref(&var_cons),
             vec![literal(integer(41)),
                  literal(integer(41)),
                  literal(integer(41))])); // too many args
}


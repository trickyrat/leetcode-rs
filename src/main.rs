pub mod solution;
use solution::*;

fn main() {
    println!("Hello, world!");
}

#[test]
fn test_two_sum() {
    assert_eq!(two_sum(vec![2, 7, 11, 15], 9), vec![0, 1]);
    assert_eq!(two_sum(vec![3, 2, 4], 6), vec![1, 2]);
    assert_eq!(two_sum(vec![3, 3], 6), vec![0, 1]);
}

#[test]
fn test_reverse_int() {
    assert_eq!(reverse_int(123), 321);
    assert_eq!(reverse_int(-123), -321);
    assert_eq!(reverse_int(120), 21);
    assert_eq!(reverse_int(100), 1);
    assert_eq!(reverse_int(2147483647), 0);
    assert_eq!(reverse_int(-2147483648), 0);
}

#[test]
fn test_complex_number_multiply() {
    assert_eq!(
        complex_number_multiply(String::from("1+1i"), String::from("1+1i")),
        String::from("0+2i")
    );
    assert_eq!(
        complex_number_multiply(String::from("1+-1i"), String::from("1+-1i")),
        String::from("0+-2i")
    );
}

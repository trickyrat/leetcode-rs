pub mod solution;
use solution::*;

fn main() {
    println!("Hello, world!");

    let mut head = ListNode::new(1);
    head.append(2);
    head.append(3);
    head.append(4);
    println!("list:{:?}", head);

    let root = generate_list_node(vec![1, 2, 3, 4]);
    println!("root:{:?}", root);
}

#[test]
fn test_two_sum() {
    assert_eq!(two_sum(vec![2, 7, 11, 15], 9), vec![0, 1]);
    assert_eq!(two_sum(vec![3, 2, 4], 6), vec![1, 2]);
    assert_eq!(two_sum(vec![3, 3], 6), vec![0, 1]);
}

#[test]
fn test_add_two_numbers() {
    assert_eq!(
        add_two_numbers(
            generate_list_node(vec![2, 4, 3]),
            generate_list_node(vec![5, 6, 4])
        ),
        generate_list_node(vec![7, 0, 8])
    );
    assert_eq!(
        add_two_numbers(generate_list_node(vec![0]), generate_list_node(vec![0])),
        generate_list_node(vec![0])
    );
    assert_eq!(
        add_two_numbers(
            generate_list_node(vec![9, 9, 9, 9, 9, 9, 9]),
            generate_list_node(vec![9, 9, 9, 9])
        ),
        generate_list_node(vec![8, 9, 9, 9, 0, 0, 0, 1])
    );
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
fn test_convert_to_base7() {
    assert_eq!(convert_to_base7(100), "202");
    assert_eq!(convert_to_base7(-7), "-10");
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

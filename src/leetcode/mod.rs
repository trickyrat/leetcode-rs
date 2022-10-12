mod data_structures;
mod interviewsolution;
mod randomizedset;
mod solution;
mod util;

pub use self::data_structures::*;
pub use self::interviewsolution::*;
pub use self::randomizedset::RandomizedSet;
pub use self::solution::*;
pub use self::util::*;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

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
                generate_list_node(vec![5, 6, 4]),
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
                generate_list_node(vec![9, 9, 9, 9]),
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
    fn test_remove_element() {
        assert_eq!(remove_element(&mut vec![3, 2, 2, 3], 3), 2);
        assert_eq!(remove_element(&mut vec![0, 1, 2, 2, 3, 0, 4, 2], 2), 5);
    }

    #[test]
    fn test_trailing_zeroes() {
        assert_eq!(trailing_zeroes(3), 0);
        assert_eq!(trailing_zeroes(5), 1);
        assert_eq!(trailing_zeroes(0), 0);
    }

    #[test]
    fn test_count_numbers_with_unique_digits() {
        assert_eq!(count_numbers_with_unique_digits(2), 91);
        assert_eq!(count_numbers_with_unique_digits(0), 1);
    }

    #[test]
    fn test_lexical_order() {
        assert_eq!(
            lexical_order(13),
            vec![1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(lexical_order(2), vec![1, 2]);
    }

    #[test]
    fn test_find_diagonal_order() {
        assert_eq!(
            find_diagonal_order(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]),
            vec![1, 2, 4, 7, 5, 3, 6, 8, 9]
        );
        assert_eq!(
            find_diagonal_order(vec![vec![1, 2], vec![3, 4]]),
            vec![1, 2, 3, 4]
        );
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

    #[test]
    fn test_exclusive_time() {
        assert_eq!(
            vec![3, 4],
            exclusive_time(
                2,
                vec!["0:start:0", "1:start:2", "1:end:5", "0:end:6"]
                    .iter()
                    .map(|&x| x.to_string())
                    .collect(),
            )
        );
        assert_eq!(
            vec![8],
            exclusive_time(
                1,
                vec![
                    "0:start:0",
                    "0:start:2",
                    "0:end:5",
                    "0:start:6",
                    "0:end:6",
                    "0:end:7",
                ]
                .iter()
                .map(|&x| x.to_string())
                .collect(),
            )
        );
        assert_eq!(
            vec![7, 1],
            exclusive_time(
                2,
                vec![
                    "0:start:0",
                    "0:start:2",
                    "0:end:5",
                    "1:start:6",
                    "1:end:6",
                    "0:end:7",
                ]
                .iter()
                .map(|&x| x.to_string())
                .collect(),
            )
        );
        assert_eq!(
            vec![8, 1],
            exclusive_time(
                2,
                vec![
                    "0:start:0",
                    "0:start:2",
                    "0:end:5",
                    "1:start:7",
                    "1:end:7",
                    "0:end:8",
                ]
                .iter()
                .map(|&x| x.to_string())
                .collect(),
            )
        );
        assert_eq!(
            vec![1],
            exclusive_time(
                1,
                vec!["0:start:0", "0:end:0"]
                    .iter()
                    .map(|&x| x.to_string())
                    .collect(),
            )
        );
    }

    #[test]
    fn test_find_longest_chain() {
        assert_eq!(
            find_longest_chain(vec![vec![1, 2], vec![2, 3], vec![3, 4]]),
            2
        );
        assert_eq!(
            find_longest_chain(vec![vec![1, 2], vec![7, 8], vec![4, 5]]),
            3
        );
    }

    #[test]
    fn test_find_closest_elements() {
        assert_eq!(
            find_closest_elements(vec![1, 2, 3, 4, 5], 4, 3),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            find_closest_elements(vec![1, 2, 3, 4, 5], 4, -1),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn test_maximum_swap() {
        assert_eq!(7236, maximum_swap(2736));
        assert_eq!(9973, maximum_swap(9973));
    }

    #[test]
    fn test_flip_lights() {
        assert_eq!(2, flip_lights(1, 1));
        assert_eq!(3, flip_lights(2, 1));
        assert_eq!(4, flip_lights(3, 1));
    }

    #[test]
    fn test_cal_points() {
        let v1 = vec!["5", "2", "C", "D", "+"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let v2 = vec!["5", "-2", "4", "C", "D", "9", "+", "+"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        assert_eq!(cal_points(v1), 30);
        assert_eq!(cal_points(v2), 27);
        assert_eq!(cal_points(vec![String::from("1")]), 1);
    }

    #[test]
    fn test_has_alternating_bits() {
        assert_eq!(has_alternating_bits(5), true);
        assert_eq!(has_alternating_bits(7), false);
        assert_eq!(has_alternating_bits(11), false);
    }

    #[test]
    fn test_pivot_index() {
        assert_eq!(
            self_dividing_numbers(1, 22),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
        );
        assert_eq!(self_dividing_numbers(47, 85), vec![48, 55, 66, 77]);
    }

    #[test]
    fn test_self_dividing_numbers() {
        assert_eq!(
            self_dividing_numbers(1, 22),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
        );
        assert_eq!(self_dividing_numbers(47, 85), vec![48, 55, 66, 77]);
    }

    #[test]
    fn test_next_greatest_letter() {
        assert_eq!(next_greatest_letter(vec!['c', 'f', 'j'], 'a'), 'c');
        assert_eq!(next_greatest_letter(vec!['c', 'f', 'j'], 'c'), 'f');
        assert_eq!(next_greatest_letter(vec!['c', 'f', 'j'], 'd'), 'f');
    }

    #[test]
    fn test_count_prime_set_bits() {
        assert_eq!(count_prime_set_bits(6, 10), 4);
        assert_eq!(count_prime_set_bits(10, 15), 5);
    }

    #[test]
    fn test_preimage_size_fzf() {
        assert_eq!(preimage_size_fzf(0), 5);
        assert_eq!(preimage_size_fzf(5), 0);
        assert_eq!(preimage_size_fzf(3), 5);
    }

    #[test]
    fn test_min_swap() {
        assert_eq!(1, min_swap(vec![1, 3, 5, 4], vec![1, 2, 3, 7]));
        assert_eq!(1, min_swap(vec![0, 3, 5, 8, 9], vec![2, 1, 4, 6, 9]));
    }

    #[test]
    fn test_unique_morse_representations() {
        let v1 = vec![
            String::from("gin"),
            String::from("zen"),
            String::from("gig"),
            String::from("msg"),
        ]
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();
        assert_eq!(unique_morse_representations(v1), 2);
        assert_eq!(unique_morse_representations(vec![String::from("a")]), 1);
    }

    #[test]
    fn test_number_of_lines() {
        assert_eq!(
            number_of_lines(
                vec![
                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                    10, 10, 10, 10, 10, 10,
                ],
                String::from("abcdefghijklmnopqrstuvwxyz"),
            ),
            vec![3, 60]
        );
        assert_eq!(
            number_of_lines(
                vec![
                    4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                    10, 10, 10, 10, 10, 10,
                ],
                String::from("bbbcccdddaaa"),
            ),
            vec![2, 4]
        );
    }

    #[test]
    fn test_subdomain_visits() {
        let cpdomains1 = vec!["9001 discuss.leetcode.com"]
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>();
        let actual1 =
            subdomain_visits(cpdomains1)
                .iter()
                .fold(HashSet::<String>::new(), |mut set, x| {
                    set.insert(x.to_string());
                    set
                });

        let cpdomains2 = vec![
            "900 google.mail.com",
            "50 yahoo.com",
            "1 intel.mail.com",
            "5 wiki.org",
        ]
        .iter()
        .map(|&x| x.to_string())
        .collect::<Vec<String>>();

        let actual2 =
            subdomain_visits(cpdomains2)
                .iter()
                .fold(HashSet::<String>::new(), |mut set, x| {
                    set.insert(x.to_string());
                    set
                });

        let expected1 = vec!["9001 discuss.leetcode.com", "9001 com", "9001 leetcode.com"]
            .iter()
            .fold(HashSet::<String>::new(), |mut set, &x| {
                set.insert(x.to_string());
                set
            });

        let expected2 = vec![
            "901 mail.com",
            "50 yahoo.com",
            "900 google.mail.com",
            "5 wiki.org",
            "5 org",
            "1 intel.mail.com",
            "951 com",
        ]
        .iter()
        .fold(HashSet::<String>::new(), |mut set, &x| {
            set.insert(x.to_string());
            set
        });

        assert_eq!(actual1, expected1);
        assert_eq!(actual2, expected2);
    }

    #[test]
    fn test_num_components() {
        let head1 = generate_list_node(vec![0, 1, 2, 3]);
        let head2 = generate_list_node(vec![0, 1, 2, 3, 4]);
        assert_eq!(2, num_components(head1, vec![0, 1, 3]));
        assert_eq!(2, num_components(head2, vec![0, 3, 1, 4]));
    }

    #[test]
    fn test_shortest_to_char() {
        assert_eq!(
            shortest_to_char(String::from("loveleetcode"), 'e'),
            vec![3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]
        );
        assert_eq!(
            shortest_to_char(String::from("aaab"), 'b'),
            vec![3, 2, 1, 0]
        );
    }

    #[test]
    fn test_unique_letter_string() {
        assert_eq!(unique_letter_string("ABC".to_string()), 10);
        assert_eq!(unique_letter_string("ABA".to_string()), 8);
        assert_eq!(unique_letter_string("LEETCODE".to_string()), 92);
    }

    #[test]
    fn test_score_of_parentheses() {
        assert_eq!(1, score_of_parentheses("()".to_string()));
        assert_eq!(2, score_of_parentheses("(())".to_string()));
        assert_eq!(2, score_of_parentheses("()()".to_string()));
    }

    #[test]
    fn test_projection_area() {
        assert_eq!(projection_area(vec![vec![1, 2], vec![3, 4]]), 17);
        assert_eq!(projection_area(vec![vec![2]]), 5);
        assert_eq!(projection_area(vec![vec![1, 0], vec![0, 2]]), 8);
    }

    #[test]
    fn test_sort_array_by_parity() {
        assert_eq!(sort_array_by_parity(vec![3, 1, 2, 4]), vec![4, 2, 1, 3]);
        assert_eq!(sort_array_by_parity(vec![0]), vec![0]);
    }

    #[test]
    fn test_min_add_to_make_valid() {
        assert_eq!(min_add_to_make_valid("())".to_string()), 1);
        assert_eq!(min_add_to_make_valid("(((".to_string()), 3);
    }

    #[test]
    fn test_three_equal_parts() {
        assert_eq!(three_equal_parts(vec![1, 0, 1, 0, 1]), vec![0, 3]);
        assert_eq!(three_equal_parts(vec![1, 1, 0, 1, 1]), vec![-1, -1]);
        assert_eq!(three_equal_parts(vec![1, 1, 0, 0, 1]), vec![0, 2]);
    }

    #[test]
    fn test_di_string_match() {
        assert_eq!(di_string_match(String::from("IDID")), vec![0, 4, 1, 3, 2]);
        assert_eq!(di_string_match(String::from("III")), vec![0, 1, 2, 3]);
        assert_eq!(di_string_match(String::from("DDI")), vec![3, 2, 0, 1]);
    }

    #[test]
    fn test_min_deletion_size() {
        let v1 = vec!["cba", "daf", "ghi"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let v2 = vec!["a", "b"].iter().map(|&x| x.to_string()).collect();
        let v3 = vec!["zyx", "wvu", "tsr"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        assert_eq!(min_deletion_size(v1), 1);
        assert_eq!(min_deletion_size(v2), 0);
        assert_eq!(min_deletion_size(v3), 3);
    }

    #[test]
    fn test_min_subsequence() {
        assert_eq!(min_subsequence(vec![4, 3, 10, 9, 8]), vec![10, 9]);
        assert_eq!(min_subsequence(vec![4, 4, 7, 6, 7]), vec![7, 7, 6]);
        assert_eq!(min_subsequence(vec![6]), vec![6]);
    }

    #[test]
    fn test_string_matching() {
        let words1: Vec<String> = vec!["mass", "as", "hero", "superhero"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let words2: Vec<String> = vec!["leetcode", "et", "code"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let words3: Vec<String> = vec!["blue", "green", "bu"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let expected1: Vec<String> = vec!["as", "hero"].iter().map(|&x| x.to_string()).collect();
        let expected2: Vec<String> = vec!["et", "code"].iter().map(|&x| x.to_string()).collect();
        let expected3: Vec<String> = vec![];
        assert_eq!(string_matching(words1), expected1);
        assert_eq!(string_matching(words2), expected2);
        assert_eq!(string_matching(words3), expected3);
    }

    #[test]
    fn test_busy_student() {
        assert_eq!(busy_student(vec![1, 2, 3], vec![3, 2, 7], 4), 1);
        assert_eq!(busy_student(vec![4], vec![4], 4), 1);
    }

    #[test]
    fn test_is_prefix_of_word() {
        assert_eq!(
            is_prefix_of_word("i love eating burger".to_string(), "burg".to_string()),
            4
        );
        assert_eq!(
            is_prefix_of_word(
                "this problem is an easy problem".to_string(),
                "pro".to_string(),
            ),
            2
        );
        assert_eq!(
            is_prefix_of_word("i am tired".to_string(), "you".to_string()),
            -1
        );
    }

    #[test]
    fn test_can_be_equal() {
        assert_eq!(can_be_equal(vec![1, 2, 3, 4], vec![2, 1, 3, 4]), true);
        assert_eq!(can_be_equal(vec![7], vec![7]), true);
        assert_eq!(can_be_equal(vec![3, 7, 9], vec![3, 7, 11]), false);
    }

    #[test]
    fn test_max_product() {
        assert_eq!(max_product(vec![3, 4, 5, 2]), 12);
        assert_eq!(max_product(vec![1, 5, 4, 5]), 16);
        assert_eq!(max_product(vec![3, 7]), 12);
    }

    #[test]
    fn test_shuffle() {
        assert_eq!(vec![2, 3, 5, 4, 1, 7], shuffle(vec![2, 5, 1, 3, 4, 7], 3));
        assert_eq!(
            vec![1, 4, 2, 3, 3, 2, 4, 1],
            shuffle(vec![1, 2, 3, 4, 4, 3, 2, 1], 4)
        );
        assert_eq!(vec![1, 2, 1, 2], shuffle(vec![1, 1, 2, 2], 2));
    }

    #[test]
    fn test_final_prices() {
        assert_eq!(vec![4, 2, 4, 2, 3], final_prices(vec![8, 4, 6, 2, 3]));
        assert_eq!(vec![1, 2, 3, 4, 5], final_prices(vec![1, 2, 3, 4, 5]));
        assert_eq!(vec![9, 0, 1, 6], final_prices(vec![10, 1, 1, 6]));
    }

    #[test]
    fn test_num_special() {
        assert_eq!(
            num_special(vec![vec![1, 0, 0], vec![0, 0, 1], vec![1, 0, 0]]),
            1
        );
        assert_eq!(
            num_special(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]),
            3
        );
    }

    #[test]
    fn test_special_array() {
        assert_eq!(special_array(vec![3, 5]), 2);
        assert_eq!(special_array(vec![0, 0]), -1);
        assert_eq!(special_array(vec![0, 4, 3, 0, 4]), 3);
    }

    #[test]
    fn test_trim_mean() {
        assert_eq!(
            (trim_mean(vec![
                1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
            ]) - 2.00000)
                <= 0.00001,
            true
        );
        assert_eq!(
            (trim_mean(vec![
                6, 2, 7, 5, 1, 2, 0, 3, 10, 2, 5, 0, 5, 5, 0, 8, 7, 6, 8, 0,
            ]) - 4.00000)
                <= 0.00001,
            true
        );
        assert_eq!(
            (trim_mean(vec![
                6, 0, 7, 0, 7, 5, 7, 8, 3, 4, 0, 7, 8, 1, 6, 8, 1, 1, 2, 4, 8, 1, 9, 5, 4, 3, 8, 5,
                10, 8, 6, 6, 1, 0, 6, 10, 8, 2, 3, 4,
            ]) - 4.77778)
                <= 0.00001,
            true
        );
    }

    #[test]
    fn test_max_length_between_equal_characters() {
        assert_eq!(0, max_length_between_equal_characters(String::from("aa")));
        assert_eq!(2, max_length_between_equal_characters(String::from("abca")));
        assert_eq!(
            -1,
            max_length_between_equal_characters(String::from("cbzyx"))
        );
    }

    #[test]
    fn test_frequency_sort() {
        assert_eq!(
            vec![3, 1, 1, 2, 2, 2],
            frequency_sort(vec![1, 1, 2, 2, 2, 3])
        );
        assert_eq!(vec![1, 3, 3, 2, 2], frequency_sort(vec![2, 3, 1, 3, 2]));
        assert_eq!(
            vec![5, -1, 4, 4, -6, -6, 1, 1, 1],
            frequency_sort(vec![-1, 1, -6, 4, 5, -6, 1, 4, 1])
        );
    }

    #[test]
    fn test_maximum_wealth() {
        assert_eq!(maximum_wealth(vec![vec![1, 2, 3], vec![3, 2, 1]]), 6);
        assert_eq!(maximum_wealth(vec![vec![1, 5], vec![7, 3], vec![3, 5]]), 10);
        assert_eq!(
            maximum_wealth(vec![vec![2, 8, 7], vec![7, 1, 3], vec![1, 9, 5]]),
            17
        );
    }

    #[test]
    fn test_reformat_number() {
        assert_eq!(
            reformat_number("1-23-45 6".to_string()),
            "123-456".to_string()
        );
        assert_eq!(
            reformat_number("123 4-567".to_string()),
            "123-45-67".to_string()
        );
        assert_eq!(
            reformat_number("123 4-5678".to_string()),
            "123-456-78".to_string()
        );
    }

    #[test]
    fn test_check_ones_segment() {
        assert_eq!(check_ones_segment("1001".to_string()), false);
        assert_eq!(check_ones_segment("110".to_string()), true);
    }

    #[test]
    fn test_are_almost_equal() {
        assert_eq!(
            true,
            are_almost_equal("bank".to_string(), "kanb".to_string())
        );
        assert_eq!(
            false,
            are_almost_equal("attack".to_string(), "defend".to_string())
        );
        assert_eq!(
            true,
            are_almost_equal("kelb".to_string(), "kelbf".to_string())
        );
    }

    #[test]
    fn test_find_the_winner() {
        assert_eq!(find_the_winner(5, 2), 3);
        assert_eq!(find_the_winner(6, 5), 1);
    }

    #[test]
    fn test_find_middle_index() {
        assert_eq!(find_middle_index(vec! {2, 3, -1, 8, 4}), 3);
        assert_eq!(find_middle_index(vec! {1, -1, 4}), 2);
        assert_eq!(find_middle_index(vec! {2, 5}), -1);
    }
}

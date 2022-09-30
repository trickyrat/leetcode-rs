use std::cmp::min;

/// 面试题 01.09. String Rotation LCCI
pub fn is_flipped_string(s1: String, s2: String) -> bool {
    if s1.is_empty() && s2.is_empty() {
        return true;
    }
    if s2.is_empty() {
        return false;
    }
    s1.repeat(2).find(&s2).is_some()
}

/// 面试题 17.09. Get Kth Magic Number LCCI
pub fn get_kth_magic_number(k: i32) -> i32 {
    let mut dp = vec![0; k as usize + 1];
    dp[1] = 1;
    let (mut p3, mut p5, mut p7) = (1, 1, 1);
    for i in 2..k as usize + 1 {
        let (num3, num5, num7) = (dp[p3] * 3, dp[p5] * 5, dp[p7] * 7);
        dp[i] = min(min(num3, num5), num7);
        if dp[i] == num3 {
            p3 += 1;
        }
        if dp[i] == num5 {
            p5 += 1;
        }
        if dp[i] == num7 {
            p7 += 1;
        }
    }
    dp[k as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_is_flipped_string() {
        assert_eq!(
            is_flipped_string("waterbottle".to_string(), "erbottlewat".to_string()),
            true
        );
        assert_eq!(
            is_flipped_string("aa".to_string(), "aba".to_string()),
            false
        );
    }

    #[test]
    fn test_get_kth_magic_number() {
        assert_eq!(get_kth_magic_number(5), 9);
    }
}

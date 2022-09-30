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
}

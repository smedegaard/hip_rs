use std::fmt;

pub trait StatusCode: fmt::Debug {
    fn is_success(&self) -> bool;
    fn code(&self) -> u32;
    fn kind_str(&self) -> &'static str;
    fn status_str(&self) -> &'static str;

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} status: {} (code: {})",
            self.kind_str(),
            self.status_str(),
            self.code()
        )
    }
}

pub trait ResultExt<T, S: StatusCode> {
    type Value;
    fn to_result(self) -> std::result::Result<Self::Value, S>;
}

impl<T, S: StatusCode> ResultExt<T, S> for (T, S) {
    type Value = T;

    fn to_result(self) -> std::result::Result<T, S> {
        let (value, status) = self;
        if status.is_success() {
            Ok(value)
        } else {
            Err(status)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestStatus {
        success: bool,
        error_code: u32,
    }

    impl StatusCode for TestStatus {
        fn is_success(&self) -> bool {
            self.success
        }

        fn code(&self) -> u32 {
            self.error_code
        }

        fn kind_str(&self) -> &'static str {
            if self.success {
                "Success"
            } else {
                "Failure"
            }
        }

        fn status_str(&self) -> &'static str {
            if self.success {
                "none"
            } else {
                "error"
            }
        }
    }

    // Implement Display for TestStatus
    #[test]
    fn test_successful_result() {
        let value = 42;
        let status = TestStatus {
            success: true,
            error_code: 0,
        };

        let result = (value, status).to_result();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_error_result() {
        let value = 42;
        let status = TestStatus {
            success: false,
            error_code: 1,
        };

        let result = (value, status).to_result();
        assert!(result.is_err());

        let err = result.err().unwrap();
        assert_eq!(err.code(), 1);
        assert!(!err.is_success());
    }

    #[test]
    fn test_status_kind_str() {
        let success_status = TestStatus {
            success: true,
            error_code: 0,
        };
        assert_eq!(success_status.kind_str(), "Success");

        let error_status = TestStatus {
            success: false,
            error_code: 1,
        };
        assert_eq!(error_status.kind_str(), "Failure");
    }

    #[test]
    fn test_status_display_format() {
        let status = TestStatus {
            success: false,
            error_code: 500,
        };
        assert_eq!(format!("{}", status), "Failure error: error (code: 500)");

        let status = TestStatus {
            success: true,
            error_code: 0,
        };
        assert_eq!(format!("{}", status), "Success error: none (code: 0)");
    }
    impl fmt::Display for TestStatus {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "{} error: {} (code: {})",
                self.kind_str(),
                if self.success { "none" } else { "error" },
                self.code()
            )
        }
    }
}

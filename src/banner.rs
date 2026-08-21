//! Terminal banner: the classic Batman ASCII logo used at startup.

/// Returns the startup banner as a plain-text string.
///
/// The banner is intentionally plain ASCII so it renders correctly both in the
/// normal terminal and inside the ratatui TUI (which strips ANSI codes).
///
/// # Returns
/// Multi-line string containing the Batman logo and a short tagline.
pub fn banner_text() -> String {
    let art = r#"           _                         _
       _==/          i     i          \==
     /XX/            |\___/|            \XX\
   /XXXX\            |XXXXX|            /XXXX\
  |XXXXXX\_         _XXXXXXX_         _/XXXXXX|
 XXXXXXXXXXXxxxxxxxXXXXXXXXXXXxxxxxxxXXXXXXXXXXX
|XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
|XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 XXXXXX/^^^^"\XXXXXXXXXXXXXXXXXXXXX/^^^^^\XXXXXX
  |XXX|       \XXX/^^\XXXXX/^^\XXX/       |XXX|
    \XX\       \X/    \XXX/    \X/       /XX/
       "\       "      \X/      "       /"
"#;
    format!("{art}\nopenBatarangs — agentic coding CLI\n")
}

/// Print the startup banner to stdout.
pub fn print_banner() {
    print!("{}", banner_text());
}

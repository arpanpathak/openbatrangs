const CYAN: &str = "\x1b[36m";
const PURPLE: &str = "\x1b[35m";
const YELLOW: &str = "\x1b[33m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

pub fn banner_text() -> String {
    let batarang = r#"
        /\                /\
       /  \              /  \
      /    \            /    \
     /      \          /      \
    <        \        /        >
     \        \      /        /
      \        \/   /        /
       \             /
        \           /
         \         /
          \       /
           \     /
            \   /
             \ /
"#;

    format!(
        "{CYAN}{BOLD}  openBatarangs{RESET}\n\
         {PURPLE}{batarang}{RESET}\n\
         {YELLOW}{BOLD}  GOTHAM CITY NIGHTS - AGENTIC CODING CLI{RESET}\n\
         {DIM}  local models - auto-discovery - no cloud{RESET}\n"
    )
}

pub fn print_banner() {
    print!("{}", banner_text());
}

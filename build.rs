use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo when to rerun this build script
    println!("cargo:rerun-if-changed=src/bindings/wrapper.hpp");
    println!("cargo:rerun-if-changed=src/bindings/native.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    // Set up HIP paths - making them configurable via environment variables
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let hip_lib_path = format!("{}/lib", rocm_path);
    let hip_include_path = format!("{}/include", rocm_path);

    // Configure library search paths and linking
    println!("cargo:rustc-link-search=native={}", hip_lib_path);
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    // Tell cargo to use hipcc as the linker, whether we're testing or not
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux" {
        // hardcode now, use `rocm_path` to build path later
        println!("cargo:rustc-linker=/opt/rocm/bin/hipcc");
        println!("cargo:rustc-link-arg=--hip-link");
    }

    // Generate bindings
    generate_bindings(&hip_include_path);

    // Compile native code
    compile_native_code(&hip_include_path);
}

fn generate_bindings(hip_include_path: &str) {
    let bindings = bindgen::Builder::default()
        .header("src/bindings/wrapper.hpp")
        .clang_arg(&format!("-I{}", hip_include_path))
        .clang_arg("-D__HIP_PLATFORM_AMD__")
        // Blocklist problematic items
        .blocklist_item("FP_INT_.*")
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")
        .blocklist_item("_Tp")
        .blocklist_item("_Value")
        // Allow HIP items
        .allowlist_type("hip.*")
        .allowlist_function("hip.*")
        // Generate proper types
        .size_t_is_usize(true)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .trust_clang_mangling(false)
        // Handle C++
        .clang_arg("-x")
        .clang_arg("c++")
        .generate()
        .expect("Unable to generate bindings");

    // Write bindings to file
    let out_path = PathBuf::from("src/bindings");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn compile_native_code(hip_include_path: &str) {
    cc::Build::new()
        .cpp(true)
        .include(hip_include_path)
        .define("__HIP_PLATFORM_AMD__", None)
        .compiler("hipcc")
        .flag("-x")
        .flag("hip")
        .file("src/bindings/native.cpp")
        .compile("native");
}

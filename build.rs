use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/bindings/wrapper.hpp");
    println!("cargo:rerun-if-changed=src/bindings/native.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    // Tell cargo to look for HIP shared libraries in /opt/rocm/lib
    println!("cargo:rustc-link-search=/opt/rocm/lib");
    // Link against the HIP runtime library
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    // Tell rustc to use hipcc for linking
    // println!("cargo:rustc-link-arg=-fgpu-rdc");
    // println!("cargo:rustc-link-arg=-hipcc");

    // Tell cargo to use hipcc as the linker
    println!("cargo:rustc-linker=hipcc");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("src/bindings/wrapper.hpp")
        // Add the HIP include path
        .clang_arg("-I/opt/rocm/include")
        // Define AMD platform (note the double underscores)
        .clang_arg("-D__HIP_PLATFORM_AMD__")
        // Only block std lib floating point constants that cause duplicates
        .blocklist_item("FP_INT_.*") // std lib floating point rounding modes
        .blocklist_item("FP_NAN") // std lib floating point categories
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")
        // Block problematic C++ template internals
        .blocklist_item("_Tp")
        .blocklist_item("_Value")
        // Allow all HIP types and functions through
        .allowlist_type("hip.*") // Allow all HIP types
        .allowlist_function("hip.*") // Allow all HIP functions
        // Generate proper types
        .size_t_is_usize(true)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .trust_clang_mangling(false)
        .clang_arg("-x")
        .clang_arg("c++")
        .generate()
        .expect("Unable to generate bindings");

    // Write bindings to file
    let out_path = PathBuf::from("src/bindings");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Compile our C++ code
    cc::Build::new()
        .cpp(true)
        .include("/opt/rocm/include")
        // Define AMD platform (note the double underscores)
        .define("__HIP_PLATFORM_AMD__", None)
        // Use the HIP compiler flags
        .compiler("hipcc") // Use the HIP compiler
        // Add HIP-specific compilation flags
        //.flag("-xhip")
        .flag("-x")
        .flag("hip")
        .flag("-fgpu-rdc")
        .file("src/bindings/native.cpp")
        .compile("native");
}

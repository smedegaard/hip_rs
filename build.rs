use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/bindings/wrapper.hpp");
    println!("cargo:rerun-if-changed=src/bindings/native.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    // Tell cargo to look for HIP shared libraries in /opt/rocm/lib
    println!("cargo:rustc-link-search=/opt/rocm/lib");
    // Link against the HIP runtime library
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("src/bindings/wrapper.hpp")
        // Add the HIP include path for bindgen
        .clang_arg("-I/opt/rocm/include")
        .clang_arg("-D__HIP_PLATFORM_AMD__")
        .trust_clang_mangling(false)
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
        // Add the HIP include path for our C++ compilation
        .include("/opt/rocm/include")
        .define("__HIP_PLATFORM_AMD", None)
        .file("src/bindings/native.cpp")
        .compile("native");
}

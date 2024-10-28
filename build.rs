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
        // Add the HIP include path
        .clang_arg("-I/opt/rocm/include")
        // Define AMD platform (note the double underscores)
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
        .include("/opt/rocm/include")
        // Define AMD platform (note the double underscores)
        .define("__HIP_PLATFORM_AMD__", None)
        // Use the HIP compiler flags
        .flag("-x hip") // Treat as HIP source
        .compiler("hipcc") // Use the HIP compiler
        .file("src/bindings/native.cpp")
        .compile("native");
}

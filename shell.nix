{pkgs ?  import <nixpkgs> {}}:
pkgs.mkShell {
  nativeBuildInputs = [
    # trival stuff
    pkgs.nix pkgs.nix-index
    pkgs.bash pkgs.less

    # algoirthm stuff
    pkgs.julia
    pkgs.openblas pkgs.liblapack pkgs.blas pkgs.lapack pkgs.mkl
    
    # scripting stuff
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.black
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pandas
    pkgs.python311Packages.matplotlib
  ];
}

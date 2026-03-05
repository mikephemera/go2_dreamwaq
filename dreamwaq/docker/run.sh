#!/bin/bash
#
# Shell script for setting my terminal environment
#
# - Author: Jongkuk Lim
# - Contact: lim.jeikei@gmail.com
# - Github: @jeikeilim

# Bash 3 does not support hash dictionary.
# hput and hget are alternative workaround.
# Usage)
# hput $VAR_NAME $KEY $VALUE
hput() {
    eval "$1""$2"='$3'
}

# Usage)
# `hget $VAR_NAME $KEY`
hget() {
    eval echo '${'"$1$2"'#hash}'
}

CMD_NAMES=(
    "arch"
    "os"
    "vim"
    "skiptmux"
    "skiprc"
    "help"
    )

hput HELP "arch" "Architecture name. Default: (auto detect. Current: $(uname -m))"
hput HELP "os" "OS name. Default: (auto detect. Current: $(uname -a | cut -d' ' -f1))"
hput HELP "vim" "Vim type to install('vim', 'nvim', 'no'). Default: 'nvim'."
hput HELP "skiptmux" "Skip install tmux. You can pass any value other than 'false' to skip. Default: 'false'"
hput HELP "skiprc" "Skip bash rc setup. You can pass any value other than 'false' to skip. Default: 'false'"
hput HELP "help" "Print this page."

hput ARGS "arch" $(uname -m)
hput ARGS "os" $(uname -a | cut -d' ' -f1)
hput ARGS "vim" "nvim"
hput ARGS "skiptmux" "false"
hput ARGS "skiprc" "false"
hput ARGS "help" "false"

if [ $# -ne 0 ]; then
    all_good=1
    while [ $# -gt 0 ]
    do
        if [ $1 == "-help" ]; then
            hput ARGS "help" "true"
            break
        fi

        # Check if the arguments are a set of '-option value'
        if [ $(echo $1 |  cut -c -1) != "-" ]; then
            all_good=0
            break
        fi

        hput ARGS "$(echo $1 | cut -c 2-)" "$2"

        shift 2

        # Check if length of the arguments are more than 2.
        if [ $# -eq 1 ]; then
            all_good=0
            break
        fi
    done

    if [ $all_good -ne 1 ]; then
        echo "Wrong usage. Arguments must be sets of '-option value ...'"
        echo "Please consider change following arguments: '$@'"
        exit 1
    fi
fi
# ----------- SETUP ARGS END -------------

OS=$(hget ARGS 'os')
ARCH=$(hget ARGS 'arch')
VIM=$(hget ARGS 'vim')
SKIP_TMUX=$(hget ARGS 'skiptmux')
SKIP_RC=$(hget ARGS 'skiprc')
PRINT_HELP=$(hget ARGS 'help')

if [ $PRINT_HELP != "false" ]; then
    echo "--------- Option Usage ($0) ---------"
    for _key in ${CMD_NAMES[@]}
    do
        echo "    -$_key value - `hget HELP $_key`"
    done
    exit 1
fi

echo "Setting on $OS-$ARCH ..."

if [ $SKIP_TMUX == "false" ]; then
    echo "Install tmux ..."

    if [ $OS == 'Linux' ]; then  # Linux-x86_64
        sudo apt -y install software-properties-common unzip build-essential tmux htop git
    elif [ $OS == 'Darwin' ]; then  # Darwin-arm64
        brew install tmux htop
    fi

    if [ -d "$HOME/.tmux" ]; then
        echo "Tmux setting already exist. Skip clone tmux setting."
    else
        git clone https://github.com/JeiKeiLim/.tmux.git ~/.tmux
        ln -s -f ~/.tmux/.tmux.conf ~/.tmux.conf
        cp ~/.tmux/.tmux.conf.local ~/.tmux.conf.local
    fi
else
    echo "Skipping install tmux."
fi


if [ $SKIP_RC == "false" ]; then
    echo "Setup bash history search"

    cat >> ~/.inputrc <<'EOF'
"\e[A": history-search-backward
"\e[B": history-search-forward
EOF

    echo "bind -f  ~/.inputrc" >> ~/.bashrc
else
    echo "Skipping setup bash history search."
fi

if [ $VIM == "nvim" ]; then
    echo "Install neovim ..."

    # Install build prerequisites and node
    if [ $OS == 'Linux' ]; then
      sudo apt-get -y install ninja-build gettext cmake unzip curl file
      sudo apt-get update
      sudo apt-get install -y ca-certificates curl gnupg
      sudo mkdir -p /etc/apt/keyrings
      curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
      
      NODE_MAJOR=18
      echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
      
      sudo apt-get update
      sudo apt-get install -y nodejs ripgrep

      # Ubuntu needs to install python virtual environment
      if [ "$(uname -a | cut -d' ' -f4 | cut -d'-' -f2)" == "Ubuntu" ]; then
        ubuntu_version=$(lsb_release -r | cut -f2 | cut -c -2)
        if [ $ubuntu_version -ge 22 ]; then
          sudo apt-get install -y python3.10-venv
        else
          sudo apt-get install -y python3.8-venv
        fi
      else
        # Non-ubuntu OS. I don't know what to do here.
        echo "Non-ubuntu OS detected. python virtual environment may need to be installed manually"
      fi
    elif [ $OS == 'Darwin' ]; then
      brew install ninja cmake gettext curl
      brew install node@18 ripgrep shellcheck
    fi

    # Build Neovim
    git clone https://github.com/neovim/neovim.git -b release-0.10
    neovim_path=$PWD/neovim
    cd neovim && make CMAKE_BUILD_TYPE=RelWithDebInfo

    # Install Neovim
    if [ $OS == 'Linux' ]; then
      cd build && cpack -G DEB && sudo dpkg -i nvim-linux64.deb
    elif [ $OS == 'Darwin' ]; then
      sudo make install
    fi
    # Clean
    cd $HOME && rm -rf $neovim_path

    sudo npm install -g neovim
    git clone https://github.com/JeiKeiLim/my_nvchad.git ~/.config/nvim
    nvim --headless "+Lazy! sync" +qa
    nvim --headless "+Lazy! sync" +qa
elif [ $VIM == 'vim' ]; then  # Linux-x86_64-Vim
    echo "Install vim ..."
    if [ $OS == 'Linux' ]; then
        sudo add-apt-repository -y ppa:jonathonf/vim
        sudo apt update
        sudo apt -y install vim
    elif [ $OS == 'Darwin' ]; then  # Darwin-arm64-Neovim
        brew install vim
    fi

    cd ~ && git clone --depth=1 https://github.com/JeiKeiLim/vimrc.git ~/.vim_runtime
    sh ~/.vim_runtime/install_awesome_vimrc.sh
    echo "Install vim ..."
else
    echo "Skipping install vim or neovim"
fi


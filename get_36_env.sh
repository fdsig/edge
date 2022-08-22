#Install Rosetta
/usr/sbin/softwareupdate --install-rosetta --agree-to-license

# Install x86_64 brew
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

# Set up x86_64 homebrew and pyenv and temporarily set aliases
alias brew86="arch -x86_64 /usr/local/bin/brew"
alias pyenv86="arch -x86_64 pyenv"

# Install required packages and flags for building this particular python version through emulation
brew86 install pyenv gcc libffi gettext
export CPPFLAGS="-I$(brew86 --prefix libffi)/include -I$(brew86 --prefix openssl)/include -I$(brew86 --prefix readline)/lib"
export CFLAGS="-I$(brew86 --prefix openssl)/include -I$(brew86 --prefix bzip2)/include -I$(brew86 --prefix readline)/include -I$(xcrun --show-sdk-path)/usr/include -Wno-implicit-function-declaration" 
export LDFLAGS="-L$(brew86 --prefix openssl)/lib -L$(brew86 --prefix readline)/lib -L$(brew86 --prefix zlib)/lib -L$(brew86 --prefix bzip2)/lib -L$(brew86 --prefix gettext)/lib -L$(brew86 --prefix libffi)/lib"

# Providing an incorrect openssl version forces a proper openssl version to be downloaded and linked during the build
export PYTHON_BUILD_HOMEBREW_OPENSSL_FORMULA=openssl@1.0

# Install Python 3.6
pyenv86 install --patch 3.6.15 <<(curl -sSL https://raw.githubusercontent.com/pyenv/pyenv/master/plugins/python-build/share/python-build/patches/3.6.15/Python-3.6.15/0008-bpo-45405-Prevent-internal-configure-error-when-runn.patch\?full_index\=1)

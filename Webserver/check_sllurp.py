import sllurp
import pkgutil

print("sllurp version:", sllurp.__version__)
print("Available submodules:")
for module in pkgutil.iter_modules(sllurp.__path__):
    print(f" - {module.name}")
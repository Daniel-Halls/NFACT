import signal
from NFACT.base.utils import colours, nprint
import warnings

warnings.filterwarnings("ignore")


class Signit_handler:
    """
    A signit handler class. Will kill. Exits
    programme safely.
    """

    def __init__(
        self,
    ) -> None:
        self.register_handler()
        self.suppress_messages = False

    def register_handler(self) -> None:
        """
        Method to registers the SIGINT handler.
        """
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self, sig, frame) -> None:
        """
        Method that handles the SIGINT signal (Ctrl+C)

        Parameters
        -----------
        sig: The signal number
        frame: The current stack frame
        """
        if not self.suppress_messages:
            col = colours()
            nprint(
                f"\n{col['darker_pink']}Recieved kill signal (Ctrl+C). Terminating..."
            )
            nprint(f"Exiting...{col['reset']}\n")
        exit(0)

    @property
    def get_suppress_messages(self):
        """
        Getter method for suppress_messages
        """
        return self.suppress_messages

    @get_suppress_messages.setter
    def set_suppress_messages(self, value: bool) -> None:
        """
        Setter for suppress_messages
        """
        self.suppress_messages = value

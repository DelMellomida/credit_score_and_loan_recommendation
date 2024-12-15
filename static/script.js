function togglePrivacyAgreement() {
  const checkbox = document.getElementById("privacy-checkbox");
  const submitButton = document.querySelector(".btn");
  const privacyAgreement = document.getElementById("privacy-agreement");

  if (checkbox.checked) {
    privacyAgreement.style.display = "block";
  } else {
    privacyAgreement.style.display = "none";
  }

  // if (checkbox.checked) {
  //   submitButton.disabled = false;
  // } else {
  //   submitButton.disabled = true;
  // }
}

// Ensure the button is initially disabled
document.addEventListener("DOMContentLoaded", function () {
  const privacyAgreement = document.getElementById("privacy-agreement");
  privacyAgreement.style.display = "none";

  // const submitButton = document.querySelector(".btn");
  // submitButton.disabled = true;

  const form = document.querySelector(".signup-form");

  form.addEventListener("submit", function (event) {
    const checkbox = document.getElementById("privacy-checkbox");
    if (!checkbox.checked) {
      alert(
        "You must agree to the Privacy Policy and Terms of Service to sign up."
      );
      event.preventDefault(); // Prevent form submission
      return false;
    }
  });
});

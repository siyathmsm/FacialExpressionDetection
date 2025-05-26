// This is an example of how to add functionality to the "Remember me" checkbox
// You can add more logic here to store and retrieve login credentials based on the checkbox state

const rememberCheckbox = document.getElementById('remember');

rememberCheckbox.addEventListener('change', function() {
  if (this.checked) {
    console.log('Remember me is checked');
    // Store login credentials here
  } else {
    console.log('Remember me is unchecked');
    // Remove login credentials here
  }
});
function goToOtherPage() {
    window.location.href = "lec_register.html";
  }
  function lecloginpage() {
    window.location.href = "lec_login.html";
  }
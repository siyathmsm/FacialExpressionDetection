document.addEventListener('DOMContentLoaded', function() {
    const mcqBtn = document.getElementById('mcq-btn');
    /* const oneWordBtn = document.getElementById('one-word-btn');
    const yesNoBtn = document.getElementById('yes-no-btn'); */
    
    const mcqForm = document.getElementById('mcq-form');
    /* const oneWordForm = document.getElementById('one-word-form');
    const yesNoForm = document.getElementById('yes-no-form'); */

    mcqBtn.addEventListener('click', function() {
        toggleForm(mcqForm);
        hideForm(oneWordForm);
        hideForm(yesNoForm);
    });

    /* oneWordBtn.addEventListener('click', function() {
        toggleForm(oneWordForm);
        hideForm(mcqForm);
        hideForm(yesNoForm);
    });

    yesNoBtn.addEventListener('click', function() {
        toggleForm(yesNoForm);
        hideForm(mcqForm);
        hideForm(oneWordForm);
    }); */

    function toggleForm(form) {
        if (form.style.display === 'none') {
            form.style.display = 'block';
        } else {
            form.style.display = 'none';
        }
    }

    function hideForm(form) {
        form.style.display = 'none';
    }
});
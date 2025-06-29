let slideIndex = 0;
let slides = document.querySelectorAll('.slide');

// Function to show the current slide
function showSlide() {
    for (let i = 0; i < slides.length; i++) {
        slides[i].classList.remove('active');
    }
    slides[slideIndex].classList.add('active');
    slideIndex = (slideIndex + 1) % slides.length;
}
setInterval(showSlide, 3000);

// Appointment form submission
document.addEventListener('DOMContentLoaded', () => {
    const appointmentForm = document.getElementById('appointmentForm');

    if (appointmentForm) {
        // Handle form submission
        appointmentForm.addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent traditional form submission

            // Collect form data
            const formData = new FormData(appointmentForm);
            const appointmentData = {
                name: formData.get('name'),
                number: formData.get('number'),
                doctor: formData.get('doctor'),
                date: formData.get('date')
            };

            // Log form data for debugging
            console.log('Form Data:', appointmentData);

            // Validate selected date
            const today = new Date().toISOString().split('T')[0];
            const selectedDate = formData.get('date');
            if (selectedDate < today) {
                alert('Please select a valid date (today or a future date).');
                return;
            }

            // Check if the user is logged in
            try {
                const sessionStatusResponse = await fetch('/session-status');
                const sessionStatus = await sessionStatusResponse.json();

                if (!sessionStatus.loggedIn) {
                    alert('You must be logged in to book an appointment.');
                    return;
                }

                // Proceed with form submission to the backend
                const response = await fetch('/appointment', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(appointmentData)
                });

                const result = await response.json();

                if (response.ok) {
                    // If the submission is successful, show success message
                    document.getElementById('successMessage').textContent = result.message;
                    appointmentForm.reset(); // Optionally, reset the form fields
                } else {
                    // If an error occurs, display the error message
                    document.getElementById('successMessage').textContent = `Error: ${result.error || 'Failed to book appointment'}`;
                }
            } catch (error) {
                // Catch any network errors
                console.error('Error submitting form:', error);
                alert('Error submitting the form. Please try again.');
            }
        });
    }
});

// Check if the user is logged in and update UI accordingly
fetch('/check-login')
    .then(response => response.json())
    .then(data => {
        const authSection = document.getElementById('auth-section');
        if (data.loggedIn) {
            // Show logout button if logged in
            authSection.innerHTML = '<button onclick="logout()">Logout</button>';
        } else {
            // Show login/signup links if not logged in
            authSection.innerHTML = '<a href="login.html">Login</a> | <a href="signup.html">Signup</a>';
        }
    });

// Logout function
function logout() {
    fetch('/logout', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.location.href = 'login.html';
            }
        });
}

// Fetch session status and display user greeting
fetch('/session-status')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.loggedIn) {
            document.getElementById('user-greeting').innerText = `Hello, ${data.user.name}`;
            document.getElementById('login-button').style.display = 'none';
            document.getElementById('logout-button').style.display = 'inline';
        } else {
            document.getElementById('user-greeting').innerText = '';
        }
    })
    .catch(error => {
        console.error('Error fetching session status:', error);
    });

// Navigation to Brain Tumor Detection Page
function goToBrainTumorPage() {
    window.location.href = "/brain-tumor-detection";
}

// Navigation to Pneumonia Detection Page
function goTopneumoniaPage() {
    window.location.href = "/pneumonia-detection";
}

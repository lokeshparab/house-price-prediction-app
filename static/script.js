// Form validation and user experience enhancements
document.getElementById('prediction-form').addEventListener('submit', function(e) {
    const submitBtn = document.querySelector('.submit-btn');
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
    submitBtn.disabled = true;
});

// Auto-calculate area validation
const areaInput = document.getElementById('area');
const pricePerSqftInput = document.getElementById('price_per_sqft');

function validateInputs() {
    const area = parseFloat(areaInput.value);
    const pricePerSqft = parseFloat(pricePerSqftInput.value);
    
    if (area && pricePerSqft) {
        const estimatedTotal = area * pricePerSqft;
        console.log('Estimated total value: ₹', estimatedTotal.toLocaleString());
    }
}

areaInput.addEventListener('input', validateInputs);
pricePerSqftInput.addEventListener('input', validateInputs);

// Add smooth animations
const formGroups = document.querySelectorAll('.form-group');
formGroups.forEach((group, index) => {
    group.style.animationDelay = `${index * 0.1}s`;
    group.style.animation = 'fadeInUp 0.6s ease forwards';
});

// CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);
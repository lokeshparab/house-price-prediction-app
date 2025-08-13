// Form animation and validation
document.getElementById('prediction-form').addEventListener('submit', function(e) {
    const submitBtn = document.querySelector('.submit-btn');
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Calculating...';
    submitBtn.disabled = true;
});

// Auto-calculate total square feet
document.addEventListener('DOMContentLoaded', function() {
    const sqftAbove = document.getElementById('sqft_above');
    const sqftBasement = document.getElementById('sqft_basement');
    
    function updateTotalSqft() {
        const above = parseInt(sqftAbove.value) || 0;
        const basement = parseInt(sqftBasement.value) || 0;
        const total = above + basement;
        
        // You can add a display element for total if needed
        console.log('Total Square Feet:', total);
    }
    
    sqftAbove.addEventListener('input', updateTotalSqft);
    sqftBasement.addEventListener('input', updateTotalSqft);
});

// Form validation and user experience improvements
const inputs = document.querySelectorAll('input[type="number"]');
inputs.forEach(input => {
    input.addEventListener('focus', function() {
        this.parentElement.style.transform = 'scale(1.02)';
    });
    
    input.addEventListener('blur', function() {
        this.parentElement.style.transform = 'scale(1)';
    });
});
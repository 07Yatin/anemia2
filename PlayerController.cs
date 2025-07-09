using UnityEngine;
[Require Component(typeof(Rigidbody))]
public class playerController:MonoBehaviour
{
    public float moveSpeed=5f;
    public float jumpForce=5f;
    private Rigidbody rb;
    private bool Grounded;
}
void Start()
{
    rb=GetComponent<Rigidbody>();
}
void Update()
{
    float h=Input.GetAxis("Horizontal");
    float v=Input.GetAxis("vertical");
    vector3 movement=new vector3(h,0,v)*movespeed;
    vector3 velocity=rb.velocity;
    rb.velocity=new vector3(movement.x,velocity.y,movement.z);
    if(Input.GetKeyDown(KeyCode.Space)&&isGrounded)
    {
        rb.AddForce(vector3.up*jumpForce,Forcemode.Impulse);
    }
    void onCollisionEnter(Collision collision)
    {
        if(collision.gameObject.CompareTag("Ground"))
        {
            isGrounded=true;
        }
    }

}